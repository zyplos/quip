// by gemini 2.5 pro
import { readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { generateImageEmbedding } from "@/transformers/USE_THESE";
import { EMBEDDING_LENGTH, IMAGES_DIR } from "@/transformers/config";

const filenames: string[] = [];
const embeddings: Buffer[] = [];

async function _generateAll() {
  const imageFiles = await readdir(IMAGES_DIR);

  // Create an array of promises, one for each image
  const embeddingPromises = imageFiles.map((file) => {
    const imagePath = join(IMAGES_DIR, file);
    return generateImageEmbedding(imagePath) as unknown as Float32Array;
  });

  // Execute all promises in parallel and wait for them to settle
  const results = await Promise.allSettled(embeddingPromises);

  // Process the results
  results.forEach((result, index) => {
    const file = imageFiles[index]; // Get the original filename using the index

    if (result.status === "fulfilled") {
      const float32Array = result.value;

      // biome-ignore lint/style/noNonNullAssertion: ok
      filenames.push(file!);
      embeddings.push(Buffer.from(float32Array.buffer));
    } else {
      // Log any images that failed to process
      console.error(`Failed to process ${file}:`, result.reason);
    }
  });

  if (embeddings.length > 0) {
    // Concatenate all individual buffers into one large buffer
    const combinedBuffer = Buffer.concat(embeddings);

    // Define output file paths
    const EMBEDDINGS_BIN_PATH = join(__dirname, "embeddings.bin");
    const EMBEDDINGS_JSON_PATH = join(__dirname, "embeddings.json");

    // Write the data to files
    await writeFile(EMBEDDINGS_BIN_PATH, combinedBuffer);
    await writeFile(
      EMBEDDINGS_JSON_PATH,
      JSON.stringify({ filenames, EMBEDDING_LENGTH }, null, 2)
    );

    console.log("✅ Embeddings saved successfully!");
    console.log(`   - Binary data: ${EMBEDDINGS_BIN_PATH}`);
    console.log(`   - Metadata: ${EMBEDDINGS_JSON_PATH}`);
  } else {
    console.log("No embeddings were generated. Check for errors above.");
  }
}

async function _generateTest() {
  const result = await generateImageEmbedding("./testImages/test.jpeg");
  console.log(result.length);
}

_generateAll();
