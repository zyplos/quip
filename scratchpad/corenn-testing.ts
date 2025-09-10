/** biome-ignore-all lint/correctness/noUnusedImports: scratchpad */
/** biome-ignore-all lint/correctness/noUnusedVariables: scratchpad */
import { EMBEDDING_LENGTH } from "@/transformers/config";
import generateImageEmbedding from "@/transformers/generateImageEmbedding";
import generateTextEmbedding from "@/transformers/generateTextEmbedding";
import { CoreNN } from "@corenn/node";
import { readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

/*
{
  create_db: [class corenn_node::create_db],
  open_db: [class corenn_node::open_db],
  new_in_memory: [class corenn_node::new_in_memory],
  insert: [class corenn_node::insert],
  query: [class corenn_node::query],
}
*/

function createDb() {
  CoreNN.create("./db", {
    // Specify the dimensionality of your vectors.
    dim: EMBEDDING_LENGTH,
    // All other config options are optional.
  });
}

async function insertDb() {
  const db = CoreNN.open("./db");

  const IMAGES_DIR = "../imagesSubset";

  const imageFiles = await readdir(IMAGES_DIR);

  // Create an array of promises, one for each image
  const embeddingPromises = imageFiles.map((file) => {
    const imagePath = join(IMAGES_DIR, file);
    return generateImageEmbedding(imagePath) as unknown as Float32Array;
  });

  // Execute all promises in parallel and wait for them to settle
  const results = await Promise.allSettled(embeddingPromises);

  type DbEntry = {
    key: string;
    vector: Float32Array;
  };

  const dbArray = results.reduce<DbEntry[]>((acc, result, index) => {
    if (result.status === "fulfilled") {
      const originalFile = imageFiles[index];

      if (originalFile) {
        // ✅ Sanitize the filename to keep only letters and numbers
        const sanitizedKey = originalFile.replace(/[^a-zA-Z0-9]/g, "");

        // console.log(sanitizedKey, result.value.length);

        acc.push({
          key: sanitizedKey, // Use the new sanitized key
          vector: result.value,
        });
      }
    }
    return acc;
  }, []);

  db.insert(dbArray);
}

// Later...
// console.log(db);
// // Array of { key, distance } objects.
// const results = CoreNN.query(db, new Float32Array([0.0, 1.1, 2.2]), 100);
// console.log(results);

async function queryDb() {
  const db = CoreNN.open("./db");
  const textEmb = (await generateTextEmbedding(
    "a cat"
  )) as unknown as Float32Array;
  const results = db.query(textEmb, 20);
  console.log(results);
}

// createDb();
// await insertDb();
await queryDb();
