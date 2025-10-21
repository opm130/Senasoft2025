"""
Agente Conversacional con Memoria Híbrida
SQLite + ChromaDB (Vectorial) + RAG
"""

import ollama
import json
import requests
import pyttsx3
import matplotlib.pyplot as plt
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from typing import Optional, Dict, Any, List
from deep_translator import GoogleTranslator
import threading
import uuid
import os
from pathlib import Path

# ============================================
# MEMORIA VECTORIAL (ChromaDB)
# ============================================

class VectorMemory:
    """Gestiona la memoria vectorial usando ChromaDB"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Usar embeddings locales (sentence-transformers)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Colección para conversaciones
        self.conversations_collection = self.client.get_or_create_collection(
            name="conversations",
            embedding_function=self.embedding_fn,
            metadata={"description": "Historial de conversaciones"}
        )
        
        # Colección para documentos (RAG)
        self.documents_collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
            metadata={"description": "Documentos cargados"}
        )
        
        print("✅ Memoria vectorial inicializada")
    
    def add_message(self, message: str, metadata: dict):
        """Agrega un mensaje a la memoria vectorial"""
        try:
            self.conversations_collection.add(
                documents=[message],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            print(f"Error al guardar en ChromaDB: {e}")
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Búsqueda semántica en conversaciones"""
        try:
            results = self.conversations_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'][0]:
                return []
            
            search_results = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                
                search_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity': 1 - distance  # Convertir distancia a similitud
                })
            
            return search_results
        except Exception as e:
            print(f"Error en búsqueda semántica: {e}")
            return []
    
    def add_document(self, content: str, metadata: dict):
        """Agrega un documento para RAG"""
        try:
            # Dividir en chunks si es muy largo
            chunks = self._split_into_chunks(content, chunk_size=500)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                
                self.documents_collection.add(
                    documents=[chunk],
                    metadatas=[chunk_metadata],
                    ids=[f"{metadata.get('filename', 'doc')}_{i}_{uuid.uuid4()}"]
                )
            
            return len(chunks)
        except Exception as e:
            print(f"Error al guardar documento: {e}")
            return 0
    
    def search_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """Busca en documentos cargados (RAG)"""
        try:
            results = self.documents_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'][0]:
                return []
            
            doc_results = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                
                doc_results.append({
                    'content': doc,
                    'source': metadata.get('filename', 'Unknown'),
                    'chunk': f"{metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
                })
            
            return doc_results
        except Exception as e:
            print(f"Error en búsqueda de documentos: {e}")
            return []
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Divide texto en chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la memoria vectorial"""
        try:
            conv_count = self.conversations_collection.count()
            doc_count = self.documents_collection.count()
            
            return {
                'conversations': conv_count,
                'document_chunks': doc_count
            }
        except:
            return {'conversations': 0, 'document_chunks': 0}


# ============================================
# MEMORIA SQL (SQLite)
# ============================================

class SQLMemory:
    """Gestiona la memoria estructurada usando SQLite"""
    
    def __init__(self, db_path="agent_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                total_messages INTEGER,
                tools_used INTEGER,
                status TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                topic TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                tool_name TEXT,
                parameters TEXT,
                result TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                upload_time TEXT,
                file_type TEXT,
                chunks_count INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (session_id, start_time, total_messages, tools_used, status)
            VALUES (?, ?, 0, 0, 'active')
        """, (session_id, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return session_id
    
    def end_session(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions 
            SET end_time = ?, status = 'completed'
            WHERE session_id = ?
        """, (datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()
    
    def save_message(self, session_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages (session_id, timestamp, role, content)
            VALUES (?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), role, content))
        
        cursor.execute("""
            UPDATE sessions 
            SET total_messages = total_messages + 1
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def save_topic(self, session_id: str, topic: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO topics (session_id, timestamp, topic)
            VALUES (?, ?, ?)
        """, (session_id, datetime.now().isoformat(), topic))
        
        conn.commit()
        conn.close()
    
    def save_tool_usage(self, session_id: str, tool_name: str, parameters: dict, result: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tool_usage (session_id, timestamp, tool_name, parameters, result)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), tool_name, json.dumps(parameters), result))
        
        cursor.execute("""
            UPDATE sessions 
            SET tools_used = tools_used + 1
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def save_document(self, filename: str, file_type: str, chunks_count: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents (filename, upload_time, file_type, chunks_count)
            VALUES (?, ?, ?, ?)
        """, (filename, datetime.now().isoformat(), file_type, chunks_count))
        
        conn.commit()
        conn.close()
    
    def get_sessions(self, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, start_time, end_time, total_messages, tools_used, status
            FROM sessions
            ORDER BY start_time DESC
            LIMIT ?
        """, (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "session_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "total_messages": row[3],
                "tools_used": row[4],
                "status": row[5]
            })
        
        conn.close()
        return sessions
    
    def get_documents(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT filename, upload_time, file_type, chunks_count
            FROM documents
            ORDER BY upload_time DESC
        """)
        
        docs = []
        for row in cursor.fetchall():
            docs.append({
                "filename": row[0],
                "upload_time": row[1],
                "file_type": row[2],
                "chunks_count": row[3]
            })
        
        conn.close()
        return docs
    
    def get_statistics(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tool_usage")
        total_tools = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT tool_name, COUNT(*) as count
            FROM tool_usage
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT 1
        """)
        most_used_tool = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_tools_used": total_tools,
            "total_documents": total_docs,
            "most_used_tool": most_used_tool[0] if most_used_tool else "N/A",
            "most_used_tool_count": most_used_tool[1] if most_used_tool else 0
        }


# ============================================
# HERRAMIENTAS
# ============================================

class ToolKit:
    def __init__(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            self.tts_available = True
        except:
            self.tts_available = False
    
    def search_web(self, query: str) -> str:
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            abstract = data.get('Abstract', '')
            if abstract:
                return f"Información encontrada: {abstract}"
            
            topics = data.get('RelatedTopics', [])
            if topics:
                results = []
                for topic in topics[:3]:
                    if 'Text' in topic:
                        results.append(topic['Text'])
                if results:
                    return "Información encontrada:\n" + "\n".join(f"- {r}" for r in results)
            
            return "No se encontró información relevante."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_weather(self, city: str = "Bogota") -> str:
        try:
            url = f"https://wttr.in/{city}?format=j1"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            current = data['current_condition'][0]
            return f"Clima en {city}: {current['temp_C']}°C, {current['weatherDesc'][0]['value']}, humedad {current['humidity']}%"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def calculate(self, expression: str) -> str:
        try:
            allowed = "0123456789+-*/(). "
            if not all(c in allowed for c in expression):
                return "Expresión no válida."
            return f"Resultado: {eval(expression)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_current_datetime(self) -> str:
        return f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def translate_text(self, text: str, target_lang: str = "en") -> str:
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return f"Traducción al {target_lang}: {translated}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_bar_chart(self, labels: list, values: list, title: str = "Gráfico de Barras") -> str:
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(labels, values, color='skyblue', edgecolor='navy')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Categorías')
            plt.ylabel('Valores')
            plt.grid(axis='y', alpha=0.3)
            
            filename = f"grafico_barras_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.show(block=False)
            plt.pause(0.1)
            
            return f"✅ Gráfico creado: {filename}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_line_chart(self, x_values: list, y_values: list, title: str = "Gráfico de Líneas") -> str:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, marker='o', linewidth=2, markersize=8, color='green')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            filename = f"grafico_lineas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.show(block=False)
            plt.pause(0.1)
            
            return f"✅ Gráfico creado: {filename}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_pie_chart(self, labels: list, values: list, title: str = "Gráfico Circular") -> str:
        try:
            plt.figure(figsize=(8, 8))
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(title, fontsize=14, fontweight='bold')
            
            filename = f"grafico_circular_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.show(block=False)
            plt.pause(0.1)
            
            return f"✅ Gráfico creado: {filename}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def speak(self, text: str) -> str:
        if not self.tts_available:
            return "TTS no disponible"
        try:
            def _speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            threading.Thread(target=_speak).start()
            return "🔊 Reproduciendo..."
        except Exception as e:
            return f"Error: {str(e)}"


# ============================================
# AGENTE CON MEMORIA HÍBRIDA
# ============================================

TOOLS_DEF = """
Herramientas: search_web, get_weather, calculate, get_current_datetime, translate_text,
create_bar_chart, create_line_chart, create_pie_chart, speak

Formato JSON: {"tool": "nombre", "parameters": {...}}
"""

class ConversationalAgent:
    def __init__(self, model="gemma3:4b", max_history=10, auto_speak=False):
        self.model = model
        self.conversation_history = []
        self.max_history = max_history
        self.toolkit = ToolKit()
        self.sql_memory = SQLMemory()
        self.vector_memory = VectorMemory()
        self.session_id = self.sql_memory.create_session()
        self.tools_used_count = 0
        self.auto_speak = auto_speak
        self.rag_mode = False
        
        self.system_prompt = f"""Eres un agente conversacional inteligente.

{TOOLS_DEF}

Usa herramientas cuando sea necesario. Responde en español naturalmente."""
        
        print(f"📝 Sesión: {self.session_id}")
    
    def add_to_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        
        # Guardar en SQL
        self.sql_memory.save_message(self.session_id, role, content)
        
        # Guardar en ChromaDB
        self.vector_memory.add_message(content, {
            "session_id": self.session_id,
            "role": role,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_messages(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Si RAG está activo y hay un mensaje del usuario, buscar en documentos
        if self.rag_mode and self.conversation_history:
            last_user_msg = next((m for m in reversed(self.conversation_history) if m['role'] == 'user'), None)
            if last_user_msg:
                doc_results = self.vector_memory.search_documents(last_user_msg['content'], 3)
                if doc_results:
                    context = "\n\n📚 CONTEXTO DE DOCUMENTOS:\n"
                    for dr in doc_results:
                        context += f"[{dr['source']} - Parte {dr['chunk']}]\n{dr['content']}\n\n"
                    
                    messages.append({"role": "system", "content": context})
        
        messages.extend(self.conversation_history)
        return messages
    
    def execute_tool(self, tool_call: Dict[str, Any]) -> str:
        tool_name = tool_call.get("tool")
        params = tool_call.get("parameters", {})
        
        try:
            if tool_name == "search_web":
                result = self.toolkit.search_web(params.get("query", ""))
            elif tool_name == "get_weather":
                result = self.toolkit.get_weather(params.get("city", "Bogota"))
            elif tool_name == "calculate":
                result = self.toolkit.calculate(params.get("expression", ""))
            elif tool_name == "get_current_datetime":
                result = self.toolkit.get_current_datetime()
            elif tool_name == "translate_text":
                result = self.toolkit.translate_text(params.get("text", ""), params.get("target_lang", "en"))
            elif tool_name == "create_bar_chart":
                result = self.toolkit.create_bar_chart(params.get("labels", []), params.get("values", []), params.get("title", "Gráfico"))
            elif tool_name == "create_line_chart":
                result = self.toolkit.create_line_chart(params.get("x_values", []), params.get("y_values", []), params.get("title", "Gráfico"))
            elif tool_name == "create_pie_chart":
                result = self.toolkit.create_pie_chart(params.get("labels", []), params.get("values", []), params.get("title", "Gráfico"))
            elif tool_name == "speak":
                result = self.toolkit.speak(params.get("text", ""))
            else:
                result = f"Herramienta no encontrada: {tool_name}"
            
            self.sql_memory.save_tool_usage(self.session_id, tool_name, params, result)
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def is_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        response = response.strip()
        
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines).strip()
        
        if response.startswith('{') and response.endswith('}'):
            try:
                data = json.loads(response)
                if "tool" in data:
                    return data
            except:
                pass
        return None
    
    def propose_topic(self):
        prompt = "Propón un tema interesante en 2-3 líneas."
        self.add_to_history("user", prompt)
        
        response = ollama.chat(model=self.model, messages=self.get_messages())
        topic = response['message']['content']
        
        self.add_to_history("assistant", topic)
        self.sql_memory.save_topic(self.session_id, topic)
        
        return topic
    
    def chat(self, user_message):
        self.add_to_history("user", user_message)
        
        try:
            response = ollama.chat(model=self.model, messages=self.get_messages())
            assistant_message = response['message']['content']
            
            tool_call = self.is_tool_call(assistant_message)
            
            if tool_call:
                print(f"🔧 {tool_call['tool']}...", end=" ", flush=True)
                tool_result = self.execute_tool(tool_call)
                print("✓")
                self.tools_used_count += 1
                
                context = f"Resultado de {tool_call['tool']}:\n{tool_result}\n\nResponde naturalmente."
                temp_msgs = self.get_messages()
                temp_msgs.append({"role": "user", "content": context})
                
                final_resp = ollama.chat(model=self.model, messages=temp_msgs)
                final_msg = final_resp['message']['content']
                self.add_to_history("assistant", final_msg)
                
                if self.auto_speak and tool_call['tool'] != 'speak':
                    self.toolkit.speak(final_msg)
                
                return final_msg
            else:
                self.add_to_history("assistant", assistant_message)
                
                if self.auto_speak:
                    self.toolkit.speak(assistant_message)
                
                return assistant_message
        except Exception as e:
            return f"Error: {str(e)}"
    
    def toggle_auto_speak(self):
        self.auto_speak = not self.auto_speak
        return f"🔊 Voz automática {'ON' if self.auto_speak else 'OFF'}"
    
    def toggle_rag_mode(self):
        self.rag_mode = not self.rag_mode
        return f"📚 Modo RAG {'ON' if self.rag_mode else 'OFF'}"
    
    def load_document(self, filepath: str) -> str:
        """Carga un documento para RAG"""
        try:
            path = Path(filepath)
            if not path.exists():
                return f"❌ Archivo no encontrado: {filepath}"
            
            # Leer contenido según tipo
            if path.suffix == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif path.suffix == '.pdf':
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(str(path))
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                except:
                    return "❌ Error al leer PDF. Instala: pip install PyPDF2"
            else:
                return f"❌ Formato no soportado: {path.suffix}. Usa .txt o .pdf"
            
            # Guardar en ChromaDB
            chunks = self.vector_memory.add_document(content, {
                'filename': path.name,
                'upload_time': datetime.now().isoformat(),
                'file_type': path.suffix
            })
            
            # Guardar metadata en SQL
            self.sql_memory.save_document(path.name, path.suffix, chunks)
            
            return f"✅ Documento cargado: {path.name} ({chunks} chunks)"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def end_session(self):
        self.sql_memory.end_session(self.session_id)
        print(f"✅ Sesión {self.session_id} guardada")


# ============================================
# INTERFAZ CLI
# ============================================

def show_sessions(sql_mem: SQLMemory):
    sessions = sql_mem.get_sessions(10)
    if not sessions:
        print("\n📭 No hay sesiones\n")
        return
    
    print("\n" + "="*70)
    print("📜 HISTORIAL DE SESIONES")
    print("="*70)
    for s in sessions:
        start = datetime.fromisoformat(s['start_time']).strftime('%Y-%m-%d %H:%M')
        status = "✓" if s['status'] == 'completed' else "⏳"
        print(f"{status} {s['session_id']} | {start} | Msgs: {s['total_messages']} | Tools: {s['tools_used']}")
    print("="*70 + "\n")

def show_documents(sql_mem: SQLMemory):
    docs = sql_mem.get_documents()
    if not docs:
        print("\n📭 No hay documentos cargados\n")
        return
    
    print("\n" + "="*70)
    print("📚 DOCUMENTOS CARGADOS")
    print("="*70)
    for doc in docs:
        time = datetime.fromisoformat(doc['upload_time']).strftime('%Y-%m-%d %H:%M')
        print(f"📄 {doc['filename']} ({doc['file_type']}) | {time} | {doc['chunks_count']} chunks")
    print("="*70 + "\n")

def semantic_search(vector_mem: VectorMemory, query: str):
    print(f"\n🔍 Buscando: '{query}'...")
    results = vector_mem.semantic_search(query, 5)
    
    if not results:
        print("No se encontraron resultados.\n")
        return
    
    print("\n" + "="*70)
    print("🧠 BÚSQUEDA SEMÁNTICA")
    print("="*70)
    for i, r in enumerate(results, 1):
        similarity = r['similarity'] * 100
        role = "👤" if r['metadata'].get('role') == 'user' else "🤖"
        time = datetime.fromisoformat(r['metadata'].get('timestamp', '')).strftime('%Y-%m-%d %H:%M')
        
        print(f"\n{i}. {role} [{time}] Similitud: {similarity:.1f}%")
        print(f"   {r['content'][:200]}{'...' if len(r['content']) > 200 else ''}")
    print("="*70 + "\n")

def show_statistics(sql_mem: SQLMemory, vector_mem: VectorMemory):
    sql_stats = sql_mem.get_statistics()
    vector_stats = vector_mem.get_stats()
    
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS GENERALES")
    print("="*70)
    print(f"Sesiones totales: {sql_stats['total_sessions']}")
    print(f"Mensajes totales: {sql_stats['total_messages']}")
    print(f"Herramientas usadas: {sql_stats['total_tools_used']}")
    print(f"Documentos cargados: {sql_stats['total_documents']}")
    print(f"Herramienta más usada: {sql_stats['most_used_tool']} ({sql_stats['most_used_tool_count']} veces)")
    print(f"\n💾 Memoria Vectorial:")
    print(f"  - Conversaciones en ChromaDB: {vector_stats['conversations']}")
    print(f"  - Chunks de documentos: {vector_stats['document_chunks']}")
    print("="*70 + "\n")

def main():
    print("="*70)
    print("🤖 AGENTE CON MEMORIA HÍBRIDA (SQL + VECTORIAL + RAG)")
    print("="*70)
    print("\nComandos disponibles:")
    print("  /proponer        - El agente propone un tema")
    print("  /resumen         - Ver resumen de conversación actual")
    print("  /voz             - Toggle voz automática")
    print("  /rag             - Toggle modo RAG (usar documentos)")
    print("  /cargar <ruta>   - Cargar documento (txt/pdf)")
    print("  /docs            - Ver documentos cargados")
    print("  /historial       - Ver últimas sesiones")
    print("  /buscar <texto>  - Búsqueda semántica en conversaciones")
    print("  /stats           - Ver estadísticas generales")
    print("  /reiniciar       - Nueva sesión")
    print("  /salir           - Salir")
    print("\n✨ Nuevas capacidades:")
    print("  🧠 Búsqueda semántica (encuentra conceptos similares)")
    print("  📚 RAG - Carga documentos y el agente responde basándose en ellos")
    print("  💾 Memoria híbrida: SQL (estructurada) + ChromaDB (vectorial)")
    print("="*70)
    
    sql_memory = SQLMemory()
    vector_memory = VectorMemory()
    agent = None
    
    print("\n🔄 Iniciando nueva sesión...\n")
    agent = ConversationalAgent(model="gemma3:4b")
    
    print("\n🤖 Agente: Déjame proponerte un tema interesante...\n")
    initial_topic = agent.propose_topic()
    print(f"🤖 Agente:\n{initial_topic}\n")
    
    while True:
        try:
            user_input = input("👤 Tú: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/salir':
                if agent:
                    agent.end_session()
                print("\n👋 ¡Hasta luego!")
                break
            
            elif user_input.lower() == '/proponer':
                print("\n🤖 Agente: Déjame pensar en un nuevo tema...\n")
                topic = agent.propose_topic()
                print(f"🤖 Agente:\n{topic}\n")
            
            elif user_input.lower() == '/resumen':
                total_msgs = len(agent.conversation_history)
                print(f"""
📊 Resumen Actual:
- Sesión: {agent.session_id}
- Mensajes: {total_msgs}
- Herramientas usadas: {agent.tools_used_count}
- Voz automática: {'✓' if agent.auto_speak else '✗'}
- Modo RAG: {'✓' if agent.rag_mode else '✗'}
""")
            
            elif user_input.lower() == '/voz':
                result = agent.toggle_auto_speak()
                print(f"\n{result}\n")
            
            elif user_input.lower() == '/rag':
                result = agent.toggle_rag_mode()
                print(f"\n{result}")
                if agent.rag_mode:
                    print("💡 Ahora el agente buscará en documentos cargados para responder\n")
                else:
                    print("💡 El agente responderá sin buscar en documentos\n")
            
            elif user_input.lower().startswith('/cargar '):
                filepath = user_input[8:].strip()
                print(f"\n📥 Cargando documento: {filepath}...")
                result = agent.load_document(filepath)
                print(f"{result}\n")
                if result.startswith('✅'):
                    print("💡 Tip: Activa modo RAG con /rag para que el agente use este documento\n")
            
            elif user_input.lower() == '/docs':
                show_documents(sql_memory)
            
            elif user_input.lower() == '/historial':
                show_sessions(sql_memory)
            
            elif user_input.lower().startswith('/buscar '):
                query = user_input[8:].strip()
                semantic_search(vector_memory, query)
            
            elif user_input.lower() == '/stats':
                show_statistics(sql_memory, vector_memory)
            
            elif user_input.lower() == '/reiniciar':
                if agent:
                    agent.end_session()
                print("\n🔄 Iniciando nueva sesión...")
                agent = ConversationalAgent(model="gemma3:4b")
                print("✅ Nueva sesión iniciada\n")
            
            else:
                print("\n🤖 Agente: ", end="", flush=True)
                response = agent.chat(user_input)
                print(f"{response}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            if agent:
                agent.end_session()
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()