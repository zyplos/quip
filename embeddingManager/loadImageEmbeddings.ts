// by gemini 2.5 pro
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export default async function loadImageEmbeddings(): Promise<
  Map<string, number[]>
> {
  // Define file paths
  const EMBEDDINGS_BIN_PATH = join(__dirname, "embeddings.bin");
  const EMBEDDINGS_JSON_PATH = join(__dirname, "embeddings.json");

  // Read the metadata and the combined buffer
  const metadataJson = await readFile(EMBEDDINGS_JSON_PATH, "utf-8");
  const { filenames, EMBEDDING_LENGTH } = JSON.parse(metadataJson);
  console.log("LOADING EMBEDDINGS, SIZE", EMBEDDING_LENGTH);
  const combinedBuffer = await readFile(EMBEDDINGS_BIN_PATH);

  // The size of one embedding in bytes (Float32 is 4 bytes per float)
  const singleEmbeddingSizeBytes = EMBEDDING_LENGTH * 4;

  const embeddingsMap = new Map<string, number[]>();

  // Slice the combined buffer to reconstruct each embedding
  for (let i = 0; i < filenames.length; i++) {
    const start = i * singleEmbeddingSizeBytes;
    const end = start + singleEmbeddingSizeBytes;

    // Create a new buffer slice for the current embedding
    const embeddingBuffer = combinedBuffer.slice(start, end);

    // Create a Float32Array view on that buffer slice
    const float32Array = new Float32Array(
      embeddingBuffer.buffer,
      embeddingBuffer.byteOffset,
      embeddingBuffer.length / 4
    );

    embeddingsMap.set(filenames[i], float32Array as unknown as number[]);
  }

  console.log(`✅ Loaded ${embeddingsMap.size} embeddings into memory.`);
  return embeddingsMap;
}

// Example usage:
// const imageEmbeddings = await loadEmbeddings();
// const embeddingForImage1 = imageEmbeddings.get('image1.jpg');
