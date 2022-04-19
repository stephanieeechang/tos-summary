from typing import List


def chunkify_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    chunkify_text splits a long text into chunks of a max of chunk_size words.
    chunkify_text tries to trim the text from the middle, i.e. it will try to
    keep two full chunks: start and end. When the text has more than 1 chunk
    size of words but less than 2, chunkfy_text will break up the text evenly,
    disregarding the chunk_size argument.
    """

    chunk_size = min(
        chunk_size, 500
    )  # bert cannot take more than 512 words, we leave some slack for tagging

    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    elif len(words) <= 2 * chunk_size:
        actual_chunk_size = len(words) // 2
        return [
            " ".join(words[:actual_chunk_size]),
            " ".join(words[actual_chunk_size:]),
        ]
    else:
        # ensure that the first and last chunks are always full sized
        first_chunk = " ".join(words[:chunk_size])
        chunks: List[str] = [first_chunk]
        last_chunk = " ".join(words[-chunk_size:])
        for start_index in range(chunk_size, len(words) - chunk_size, chunk_size):
            end_index = min(len(words) - chunk_size, start_index + chunk_size)
            chunks.append(" ".join(words[start_index:end_index]))
        chunks.append(last_chunk)
        return chunks
