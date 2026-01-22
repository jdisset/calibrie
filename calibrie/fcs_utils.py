# Copyright (c) 2026 Jean Disset
# MIT License - see LICENSE file for details.

from __future__ import annotations

import numpy as np


def escape_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_").upper()


def _parse_int(value: str) -> int:
    try:
        return int(value.strip())
    except Exception:
        return -1


def _bytes_from_pnb(value: str) -> int:
    raw = _parse_int(value)
    if raw <= 0:
        return raw
    # Some files store PnB in bits (e.g. 32 for 4-byte float)
    if raw > 8 and raw % 8 == 0:
        return raw // 8
    return raw


def _parse_header(file_bytes: bytes) -> tuple[int, int, int, int]:
    if len(file_bytes) < 58:
        raise ValueError("Invalid FCS header (too short).")
    header = file_bytes[:58].decode("ascii", errors="ignore")
    text_start = _parse_int(header[10:18])
    text_end = _parse_int(header[18:26])
    data_start = _parse_int(header[26:34])
    data_end = _parse_int(header[34:42])
    return text_start, text_end, data_start, data_end


def _parse_text_segment(segment: bytes) -> dict[str, str]:
    if not segment:
        return {}
    delim = segment[:1]
    if not delim:
        return {}

    parts: list[str] = []
    buf = bytearray()
    i = 1
    seg_len = len(segment)
    while i < seg_len:
        b = segment[i:i + 1]
        if b == delim:
            if i + 1 < seg_len and segment[i + 1:i + 2] == delim:
                buf.append(delim[0])
                i += 2
                continue
            parts.append(buf.decode("utf-8", errors="ignore"))
            buf.clear()
            i += 1
            continue
        buf.append(segment[i])
        i += 1
    if buf:
        parts.append(buf.decode("utf-8", errors="ignore"))

    meta: dict[str, str] = {}
    for i in range(0, len(parts) - 1, 2):
        key = parts[i]
        value = parts[i + 1]
        if key:
            meta[key] = value
    return meta


def parse_fcs_meta(file_bytes: bytes) -> dict[str, str]:
    text_start, text_end, data_start, data_end = _parse_header(file_bytes)
    if text_start < 0 or text_end < text_start:
        raise ValueError("Invalid FCS header (text segment).")

    text_segment = file_bytes[text_start:text_end + 1]
    meta = _parse_text_segment(text_segment)

    # Prefer TEXT segment data positions if present
    begindata = meta.get("$BEGINDATA")
    enddata = meta.get("$ENDDATA")
    if begindata is not None and enddata is not None:
        data_start = _parse_int(begindata)
        data_end = _parse_int(enddata)
        meta["$BEGINDATA"] = str(data_start)
        meta["$ENDDATA"] = str(data_end)
    else:
        meta["$BEGINDATA"] = str(data_start)
        meta["$ENDDATA"] = str(data_end)

    return meta


def parse_fcs_data(file_bytes: bytes, report_progress=None) -> dict:
    import numpy as np
    if report_progress:
        report_progress(0.05, "Parsing header...")

    meta = parse_fcs_meta(file_bytes)

    par = _parse_int(meta.get("$PAR", "0"))
    point_count = _parse_int(meta.get("$TOT", "0"))
    if par <= 0:
        raise ValueError("Invalid FCS metadata ($PAR missing).")

    channels = [escape_name(meta.get(f"$P{i}N", f"P{i}")) for i in range(1, par + 1)]

    progress_milestones = None
    if report_progress:
        report_progress(0.2, "Parsing metadata...")
        report_progress(0.3, "Decoding events (0%)...")
        progress_milestones = [i / 10 for i in range(1, 11)]

    begin = _parse_int(meta.get("$BEGINDATA", "-1"))
    end = _parse_int(meta.get("$ENDDATA", "-1"))
    if begin < 0 or end < begin:
        raise ValueError("Invalid FCS metadata (data segment not found).")

    dtype = meta.get("$DATATYPE", "F").upper()
    byteord = meta.get("$BYTEORD", "1,2,3,4")
    bytes_per = _bytes_from_pnb(meta.get("$P1B", "4"))
    same_bytes = all(_bytes_from_pnb(meta.get(f"$P{i}B", str(bytes_per))) == bytes_per for i in range(1, par + 1))
    if not same_bytes:
        raise ValueError("Unsupported FCS: varying parameter byte sizes.")

    if dtype == "F":
        if bytes_per == 4:
            base_dtype = "f4"
        elif bytes_per == 8:
            base_dtype = "f8"
        else:
            raise ValueError(f"Unsupported float width: {bytes_per} bytes.")
    elif dtype == "I":
        if bytes_per == 1:
            base_dtype = "u1"
        elif bytes_per == 2:
            base_dtype = "u2"
        elif bytes_per == 4:
            base_dtype = "u4"
        else:
            raise ValueError(f"Unsupported int width: {bytes_per} bytes.")
    elif dtype == "D":
        base_dtype = "f8"
    else:
        raise ValueError(f"Unsupported FCS datatype: {dtype}")

    endian = "<" if "1,2,3,4" in byteord else ">"
    dtype_np = np.dtype(f"{endian}{base_dtype}")
    bytes_per_event = par * bytes_per

    raw = memoryview(file_bytes)[begin:end + 1]
    max_rows = len(raw) // bytes_per_event
    if point_count <= 0:
        point_count = max_rows
    rows_total = min(max_rows, point_count) if point_count > 0 else max_rows
    if rows_total < max_rows:
        raw = raw[:rows_total * bytes_per_event]

    if dtype == "F" and bytes_per == 4 and endian == "<":
        if report_progress:
            report_progress(0.35, "Decoding events (fast path)...")
        arr = np.frombuffer(raw, dtype=dtype_np, count=rows_total * par)
        if arr.size >= rows_total * par:
            arr = arr[: rows_total * par]
        arr = arr.reshape((rows_total, par))
        data = arr.astype(np.float32, copy=False)
        if report_progress and progress_milestones:
            for pct in progress_milestones:
                report_progress(0.3 + 0.6 * pct, f"Decoding events ({int(pct * 100)}%)...")
        if report_progress:
            report_progress(0.9, "Packing data...")
        packed = {
            "data_bytes": data.tobytes(),
            "shape": [int(data.shape[0]), int(data.shape[1])],
        }
        if report_progress:
            report_progress(1.0, "Finalizing...")
        return {
            "point_count": point_count,
            "channels": channels,
            "data_packed": packed,
        }

    chunk_rows = 100_000
    data = None
    write_offset = 0
    processed_rows = 0

    milestone_idx = 0
    for start_row in range(0, rows_total, chunk_rows):
        end_row = min(rows_total, start_row + chunk_rows)
        chunk = raw[start_row * bytes_per_event:end_row * bytes_per_event]
        if not chunk:
            break
        arr = np.frombuffer(chunk, dtype=dtype_np)
        rows = arr.size // par
        if rows <= 0:
            continue
        arr = arr[:rows * par].reshape((rows, par))
        if endian == ">":
            arr = arr.byteswap().newbyteorder()
        arr = arr.astype(np.float32, copy=False)
        if data is None:
            data = np.empty((max_rows, par), dtype=np.float32)
        end_write = write_offset + arr.shape[0]
        data[write_offset:end_write, :] = arr
        write_offset = end_write
        processed_rows = min(rows_total, end_row)
        if report_progress and rows_total > 0:
            pct = processed_rows / rows_total
            while progress_milestones and milestone_idx < len(progress_milestones) and pct >= progress_milestones[milestone_idx]:
                milestone_pct = progress_milestones[milestone_idx]
                report_progress(0.3 + 0.6 * milestone_pct, f"Decoding events ({int(milestone_pct * 100)}%)...")
                milestone_idx += 1

    if report_progress:
        report_progress(0.95, "Packing data...")

    if data is None:
        data = np.zeros((0, par), dtype=np.float32)
    else:
        data = data[:write_offset]
    packed = {
        "data_bytes": data.tobytes(),
        "shape": [int(data.shape[0]), int(data.shape[1])],
    }

    if report_progress:
        report_progress(1.0, "Finalizing...")

    return {
        "point_count": point_count,
        "channels": channels,
        "data_packed": packed,
    }
