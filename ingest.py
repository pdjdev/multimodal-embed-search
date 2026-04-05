import argparse
import json
import math
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import sqlite_vec
from tqdm import tqdm


CHUNK_RE = re.compile(r"^(.+?)_(\d+)\.mp4$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Sync chopped clips and ingest Gemini embeddings.")
    parser.add_argument("--force-reembed", action="store_true", help="Re-embed all chopped clips.")
    return parser.parse_args()


def load_settings():
    load_dotenv()
    return {
        'videos_dir': Path(os.getenv("VIDEOS_DIR", "videos")),
        'chopped_dir': Path(os.getenv("CHOPPED_DIR", "chopped")),
        'db_path': Path("db/video_index.db"),
        'chunk_length': int(os.getenv("CHUNK_LENGTH", "32")),
        'chunk_overlap': int(os.getenv("CHUNK_OVERLAP", "8")),
        'ffmpeg_encoder': os.getenv("FFMPEG_ENCODER", "libx264").strip(),
        'ffmpeg_preset': os.getenv("FFMPEG_PRESET", "veryfast").strip(),
        'ffmpeg_crf': int(os.getenv("FFMPEG_CRF", "28")),
        'ffmpeg_audio_codec': os.getenv("FFMPEG_AUDIO_CODEC", "aac").strip(),
        'ffmpeg_audio_bitrate': os.getenv("FFMPEG_AUDIO_BITRATE", "96k").strip(),
        'embed_model': os.getenv("EMBED_MODEL", "gemini-embedding-2-preview"),
        'embed_dim': int(os.getenv("EMBED_DIM", "768")),
        'video_extensions': tuple(ext.strip().lower() for ext in os.getenv("VIDEO_EXTENSIONS", ".mp4,.mov,.mkv,.webm").split(",")),
    }


def ensure_dirs(settings):
    settings['videos_dir'].mkdir(parents=True, exist_ok=True)
    settings['chopped_dir'].mkdir(parents=True, exist_ok=True)
    settings['db_path'].parent.mkdir(parents=True, exist_ok=True)


def collect_source_videos(settings):
    videos = {}
    for path in settings['videos_dir'].iterdir():
        if path.is_file() and path.suffix.lower() in settings['video_extensions']:
            videos[path.stem] = path
    return videos


def collect_chopped_chunks(chopped_dir):
    grouped = {}
    for path in chopped_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        m = CHUNK_RE.match(path.name)
        if m:
            stem = m.group(1)
            grouped.setdefault(stem, []).append(path)
    
    for chunks in grouped.values():
        chunks.sort(key=chunk_sort_key)
    return grouped


def chunk_sort_key(path):
    m = CHUNK_RE.match(path.name)
    return int(m.group(2)) if m else -1


def get_video_duration(input_file):
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(input_file)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    duration = float(json.loads(proc.stdout).get("format", {}).get("duration", 0))
    if duration <= 0:
        raise RuntimeError(f"Invalid duration: {input_file.name}")
    return duration





def build_chunk_plan(source_video, chopped_dir, chunk_length, overlap):
    total_duration = get_video_duration(source_video)
    stride = chunk_length - overlap
    num_chunks = max(1, math.ceil((total_duration - overlap) / stride))
    
    plan = []
    for idx in range(num_chunks):
        start = idx * stride
        end = min(start + chunk_length, total_duration)
        plan.append({
            'source_name': source_video.stem,
            'chunk_index': idx,
            'chunk_path': chopped_dir / f"{source_video.stem}_{idx:04d}.mp4",
            'start_time': float(start),
            'end_time': float(end),
        })
    return plan


def run_cmd(cmd):
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        if result.stderr:
            print(f"[STDERR]\n{result.stderr}")
        if result.stdout:
            print(f"[STDOUT]\n{result.stdout}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)


def build_ffmpeg_encode_cmd(source_video, chunk, chunk_length, settings):
    encoder = settings['ffmpeg_encoder'].lower()
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(source_video),
        "-ss", f"{chunk['start_time']:.3f}",
        "-t", str(chunk_length),
        "-map", "0:v:0", "-map", "0:a?",        
        "-c:v", settings['ffmpeg_encoder'],
    ]
    
    # QSV 전용 파라미터
    if 'qsv' in encoder:
        cmd.extend(["-global_quality:v", str(settings['ffmpeg_crf']), "-preset:v", settings['ffmpeg_preset']])  # QSV: q값 (0-51, 높을수록 빠름)
        cmd.extend(["-vf", "scale=-1:360"]) # 높이 360p
    else:
        cmd.extend(["-crf", str(settings['ffmpeg_crf']), "-preset", settings['ffmpeg_preset']])
        cmd.extend(["-vf", "scale=-2:360"]) # 높이 360p
    
    cmd.extend([
        "-c:a", settings['ffmpeg_audio_codec'],
        "-b:a", settings['ffmpeg_audio_bitrate'],
        str(chunk['chunk_path']),
    ])
    return cmd


def encode_chunk(source_video, chunk, chunk_length, settings):
    run_cmd(build_ffmpeg_encode_cmd(source_video, chunk, chunk_length, settings))





def chop_video(source_video, settings):
    total_duration = get_video_duration(source_video)
    plan = build_chunk_plan(source_video, settings['chopped_dir'], settings['chunk_length'], settings['chunk_overlap'])
    display_name = ellipsis_middle(source_video.name)
    
    for p in settings['chopped_dir'].glob(f"{source_video.stem}_*.mp4"):
        p.unlink(missing_ok=True)
    
    for chunk in tqdm(plan, desc=f"청크 생성 {display_name}", unit="청크"):
        encode_chunk(source_video, chunk, settings['chunk_length'], settings)


def chunk_layout_is_current(settings, source_video, existing_chunks):
    """기존 청크 파일이 있고 현재 설정과 맞으면 True를 반환."""
    if not existing_chunks:
        return False
    
    expected = build_chunk_plan(source_video, settings['chopped_dir'], settings['chunk_length'], settings['chunk_overlap'])
    actual_count = len(sorted(existing_chunks, key=chunk_sort_key))
    expected_count = len(expected)
    return actual_count == expected_count


def sync_chopped(settings):
    source_videos = collect_source_videos(settings)
    chopped_grouped = collect_chopped_chunks(settings['chopped_dir'])
    
    stale_stems = set(chopped_grouped.keys()) - set(source_videos.keys())
    for stem in stale_stems:
        for chunk in chopped_grouped[stem]:
            chunk.unlink(missing_ok=True)
        print(f"[sync] removed stale chunks for '{stem}'")
    
    for stem, video_path in source_videos.items():
        current_chunks = chopped_grouped.get(stem, [])
        if current_chunks and chunk_layout_is_current(settings, video_path, current_chunks):
            continue
        print(f"[sync] chopping {video_path.name} (L={settings['chunk_length']}s, O={settings['chunk_overlap']}s)")
        chop_video(video_path, settings)


def open_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(conn, dim):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks ("
        "id INTEGER PRIMARY KEY, source_name TEXT NOT NULL, chunk_index INTEGER NOT NULL, "
        "chunk_path TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);"
    )
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding FLOAT[{dim}]);")
    conn.commit()


def parse_chunk_meta(chunk_path):
    m = CHUNK_RE.match(chunk_path.name)
    if not m:
        raise ValueError(f"Invalid chunk file name: {chunk_path.name}")
    return {'source_name': m.group(1), 'chunk_index': int(m.group(2))}


def sync_db_rows_to_files(conn, chopped_dir):
    existing = set(str(p.as_posix()) for p in chopped_dir.glob("*.mp4"))
    rows = conn.execute("SELECT id, chunk_path FROM chunks").fetchall()
    stale_ids = [row_id for row_id, chunk_path in rows if chunk_path not in existing]
    
    if stale_ids:
        marks = ",".join("?" for _ in stale_ids)
        conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({marks})", stale_ids)
        conn.execute(f"DELETE FROM chunks WHERE id IN ({marks})", stale_ids)
        conn.commit()
        print(f"[db] removed {len(stale_ids)} stale rows")


def embed_video_file(client, model, dim, path):
    with path.open("rb") as f:
        payload = f.read()
    result = client.models.embed_content(
        model=model,
        contents=[types.Part.from_bytes(data=payload, mime_type="video/mp4")],
        config=types.EmbedContentConfig(output_dimensionality=dim),
    )
    return list(result.embeddings[0].values)


def has_vector(conn, row_id):
    return conn.execute("SELECT 1 FROM vec_chunks WHERE rowid = ? LIMIT 1", (row_id,)).fetchone() is not None


def ellipsis_middle(text, max_len=22):
    if len(text) <= max_len:
        return text
    keep = max_len - 3
    left = max(4, keep // 2)
    return f"{text[:left]}...{text[-(keep - left):]}"


def ingest_chunks(settings, force_reembed):
    client = genai.Client()
    conn = open_db(settings['db_path'])
    try:
        init_db(conn, settings['embed_dim'])
        sync_db_rows_to_files(conn, settings['chopped_dir'])
        
        chunks = sorted(settings['chopped_dir'].glob("*.mp4"), key=lambda p: (parse_chunk_meta(p)['source_name'], chunk_sort_key(p)))
        
        # Step 1: 모든 청크를 DB에 먼저 삽입(upsert)
        print(f"[embed] {len(chunks)}개 청크 DB 등록 중...")
        for chunk in chunks:
            meta = parse_chunk_meta(chunk)
            conn.execute(
                "INSERT INTO chunks(source_name, chunk_index, chunk_path) VALUES (?, ?, ?) "
                "ON CONFLICT(chunk_path) DO UPDATE SET source_name=excluded.source_name, chunk_index=excluded.chunk_index",
                (meta['source_name'], meta['chunk_index'], chunk.as_posix()),
            )
        conn.commit()
        
        # Step 2: 임베딩이 필요한 청크만 필터링
        need_embedding = []
        for chunk in chunks:
            row_id = conn.execute("SELECT id FROM chunks WHERE chunk_path = ?", (chunk.as_posix(),)).fetchone()[0]
            if force_reembed or not has_vector(conn, row_id):
                need_embedding.append(chunk)
        
        skipped = len(chunks) - len(need_embedding)
        if skipped > 0:
            print(f"[embed] 기존 임베딩 감지: {skipped}개 스킵")
        
        # Step 3: 필요한 청크들만 임베딩 (프로그레시브 바는 실제 작업만 표시)
        embedded = 0
        for chunk in tqdm(need_embedding, desc="임베딩 진행", unit="청크"):
            row_id = conn.execute("SELECT id FROM chunks WHERE chunk_path = ?", (chunk.as_posix(),)).fetchone()[0]
            embedding = embed_video_file(client, settings['embed_model'], settings['embed_dim'], chunk)
            packed = sqlite_vec.serialize_float32(embedding)
            conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (row_id,))
            conn.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (row_id, packed))
            conn.commit()
            embedded += 1
        
        print(f"[embed] 완료 - 신규 임베딩: {embedded}, 기존: {skipped}, 총합: {len(chunks)}")
    finally:
        conn.close()


def main():
    args = parse_args()
    settings = load_settings()
    ensure_dirs(settings)
    sync_chopped(settings)
    ingest_chunks(settings, force_reembed=args.force_reembed)
    print("[done] ingest complete")


if __name__ == "__main__":
    main()
