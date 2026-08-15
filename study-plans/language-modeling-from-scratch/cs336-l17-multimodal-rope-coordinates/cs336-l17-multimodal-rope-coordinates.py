import torch

def multimodal_rope_coordinates(segments):
    times = []
    rows = []
    cols = []

    offset = 0

    for segment in segments:
        seg_type = segment["type"]

        if seg_type == "text":
            length = segment["length"]
            for j in range(length):
                pos = offset + j
                times.append(pos)
                rows.append(pos)
                cols.append(pos)
            offset += length

        elif seg_type == "separator":
            times.append(offset)
            rows.append(offset)
            cols.append(offset)
            offset += 1

        elif seg_type == "image":
            height = segment["height"]
            width = segment["width"]
            for r in range(height):
                for c in range(width):
                    times.append(offset)
                    rows.append(offset + r)
                    cols.append(offset + c)
            if height > 0 and width > 0:
                offset += max(height, width)
            else:
                offset += max(height, width)  # covers zero-size edge case too

        elif seg_type == "video":
            frames = segment["frames"]
            height = segment["height"]
            width = segment["width"]
            stride = segment["frame_stride"]
            for f in range(frames):
                for r in range(height):
                    for c in range(width):
                        times.append(offset + f * stride)
                        rows.append(offset + r)
                        cols.append(offset + c)
            if frames > 0 and height > 0 and width > 0:
                offset += max((frames - 1) * stride + 1, height, width)
            else:
                offset += max((frames - 1) * stride + 1 if frames > 0 else 0, height, width)

        else:
            raise ValueError(f"Unknown segment type: {seg_type}")

    coordinates = torch.tensor([times, rows, cols], dtype=torch.int64, device="cpu")

    return {"coordinates": coordinates}