import os
import sqlite3
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import sqlite_vec
from rich.console import Console
from rich.table import Table
from rich import box


def load_config():
    load_dotenv()
    return {
        'db_path': Path("db/video_index.db"),
        'model': os.getenv("EMBED_MODEL", "gemini-embedding-2-preview"),
        'dim': int(os.getenv("EMBED_DIM", "768")),
        'top_k': int(os.getenv("SEARCH_TOP_K", "5")),
    }


def open_db(path):
    if not path.exists():
        raise FileNotFoundError(f"DB 파일 없음: {path}. ingest.py 실행하세요.")
    
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def embed_query_text(client, model, dim, text):
    result = client.models.embed_content(
        model=model,
        contents=[text],
        config={"output_dimensionality": dim},
    )
    return sqlite_vec.serialize_float32(list(result.embeddings[0].values))


def search(conn, query_embedding, top_k):
    rows = conn.execute(
        "SELECT chunks.chunk_path, vec_chunks.distance FROM vec_chunks "
        "JOIN chunks ON chunks.id = vec_chunks.rowid "
        "WHERE vec_chunks.embedding MATCH ? AND k = ? "
        "ORDER BY vec_chunks.distance ASC",
        (query_embedding, top_k),
    ).fetchall()
    return [(p, float(d)) for p, d in rows]


def truncate_middle(text, max_len):
    if len(text) <= max_len:
        return text
    keep = max_len - 3
    left = keep // 2
    return f"{text[:left]}...{text[-(keep - left):]}"


def print_results_table(results):
    console = Console()
    
    table = Table(
        title="Search Results",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
    )
    
    table.add_column("No", style="cyan", width=5)
    table.add_column("File Path", style="green")
    table.add_column("Distance", style="yellow", width=12)
    
    for idx, (path, distance) in enumerate(results, start=1):
        rel = f"./{Path(path).as_posix()}"
        table.add_row(
            str(idx),
            rel,
            f"{distance:.6f}"
        )
    
    console.print(table)


def play_video_clip(path):
    if not path.exists():
        print(f"File not found: {path.as_posix()}")
        return

    vlc_path = next(
        (p for p in [
            os.getenv("VLC_BIN"),
            r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
        ] if p and Path(p).exists()),
        None
    )
    
    if not vlc_path:
        print("VLC not found. Install VLC or set VLC_BIN in .env")
        return
    
    try:
        subprocess.Popen([vlc_path, str(path.resolve())])
        print(f"Playing: {path.name}")
    except Exception as e:
        print(f"Playback error: {e}")


def run_repl(conn, client, model, dim, top_k):
    print("Search mode (exit: /exit or Ctrl+C)")
    last_results = []
    
    while True:
        cmd = input("search> ").strip()
        if not cmd:
            continue

        if cmd.lower() in {"/exit", "/quit", "exit", "quit"}:
            break

        if cmd.startswith("/") and cmd[1:].isdigit():
            rank = int(cmd[1:])
            if 1 <= rank <= len(last_results):
                path = Path(last_results[rank - 1][0])
                play_video_clip(path)
            else:
                print(f"Enter 1 to {len(last_results)}")
            continue

        emb = embed_query_text(client, model, dim, cmd)
        results = search(conn, emb, top_k)
        last_results = results
        
        if results:
            print_results_table(results)
        else:
            print("No results found")


def main():
    cfg = load_config()
    client = genai.Client()
    conn = open_db(cfg['db_path'])
    try:
        run_repl(conn, client, cfg['model'], cfg['dim'], cfg['top_k'])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
