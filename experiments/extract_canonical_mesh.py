"""Recover MediaPipe's canonical face mesh from the face_landmarker.task bundle.

The bundle shipped in the A3R repository (unity_project/Assets/StreamingAssets/
face_landmarker.task) is a zip archive whose member
``geometry_pipeline_metadata_landmarks.binarypb`` is a serialized
``FaceGeometryPipelineMetadata`` protobuf.  Its ``canonical_mesh`` (a Mesh3d)
stores a packed float ``vertex_buffer`` with 5 floats per vertex (x, y, z, u, v)
for 468 vertices, in centimetres, in the same vertex order as the FaceLandmarker
landmark topology (indices 0-467; iris landmarks 468-477 are not included).

This script uses a minimal protobuf wire-format reader (no protobuf dependency)
and writes ``canonical_face_468.csv`` (index, x_cm, y_cm, z_cm).

Usage: python extract_canonical_mesh.py <face_landmarker.task> <out_csv>
"""
import struct
import sys
import zipfile
import io
import csv


def read_varint(buf, pos):
    result, shift = 0, 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def iter_fields(buf):
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field, wtype = key >> 3, key & 7
        if wtype == 0:
            val, pos = read_varint(buf, pos)
        elif wtype == 1:
            val = buf[pos:pos + 8]; pos += 8
        elif wtype == 2:
            ln, pos = read_varint(buf, pos)
            val = buf[pos:pos + ln]; pos += ln
        elif wtype == 5:
            val = buf[pos:pos + 4]; pos += 4
        else:
            raise ValueError(f"unsupported wire type {wtype}")
        yield field, wtype, val


def find_vertex_buffer(buf):
    """Return the canonical vertex buffer.

    FaceGeometryPipelineMetadata field 1 is ``canonical_mesh`` (Mesh3d); within
    it, field 3 is ``vertex_buffer`` (repeated float).  In this bundle the
    floats are serialized unpacked (wire type 5, one fixed32 per value); a
    packed encoding (wire type 2) is also handled.
    """
    for field, wtype, val in iter_fields(buf):
        if field == 1 and wtype == 2:
            floats = []
            for f2, w2, v2 in iter_fields(val):
                if f2 == 3 and w2 == 5:
                    floats.append(struct.unpack("<f", v2)[0])
                elif f2 == 3 and w2 == 2:
                    floats.extend(struct.unpack(f"<{len(v2) // 4}f", v2))
            if len(floats) == 468 * 5:
                return floats
    return None


def main(task_path, out_csv):
    with zipfile.ZipFile(task_path) as z:
        data = z.read("geometry_pipeline_metadata_landmarks.binarypb")
    vb = find_vertex_buffer(data)
    if vb is None:
        raise SystemExit("canonical vertex buffer not found")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "x_cm", "y_cm", "z_cm"])
        for i in range(468):
            x, y, z = vb[5 * i:5 * i + 3]
            w.writerow([i, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
    print(f"wrote {out_csv} (468 vertices)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
