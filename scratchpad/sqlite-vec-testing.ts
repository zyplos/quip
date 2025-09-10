/** biome-ignore-all lint/correctness/noUnusedImports: scratchpad */
/** biome-ignore-all lint/correctness/noUnusedVariables: scratchpad */
import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import generateImageEmbedding from "@/transformers/generateImageEmbedding";
import generateTextEmbedding from "@/transformers/generateTextEmbedding";
import { readdir } from "node:fs/promises";
import { join } from "node:path";

const db = new Database(":memory:");
sqliteVec.load(db);

db.run("CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[768])");

const insertStmt = db.prepare(
  "INSERT INTO vec_items(rowid, embedding) VALUES (?, vec_f32(?))"
);

const insertVectors = db.transaction((items) => {
  for (const [id, vector] of items) {
    insertStmt.run(BigInt(id), new Float32Array(vector));
  }
});

////////////////////

const IMAGES_DIR = "../imagesSubset";

const imageFiles = await readdir(IMAGES_DIR);

// Create an array of promises, one for each image
const embeddingPromises = imageFiles.map((file) => {
  const imagePath = join(IMAGES_DIR, file);
  return generateImageEmbedding(imagePath) as unknown as Float32Array;
});

// Execute all promises in parallel and wait for them to settle
const results = await Promise.allSettled(embeddingPromises);

type DbInsertDocument = [key: number, embedding: Float32Array];

const MAIN_ITEMS: DbInsertDocument[] = [];
results.forEach((result, index) => {
  if (result.status === "fulfilled") {
    const originalFile = imageFiles[index];

    if (originalFile) {
      MAIN_ITEMS.push([index, result.value]);
    }
  }
});

insertVectors(MAIN_ITEMS);

////////////////////

type DbRow = {
  rowid: number;
  distance: number;
};

const QUERY = await generateTextEmbedding("laptop");

const rows = db
  .prepare(
    `
  SELECT
    rowid,
    distance
  FROM vec_items
  WHERE embedding MATCH ?
  ORDER BY distance
  LIMIT 30
`
  )
  .all(new Float32Array(QUERY)) as DbRow[];

const labeledRows = rows.map(({ rowid, distance }) => {
  return {
    fileName: `C:/devspace/backend/quip/imagesSubset/${imageFiles[rowid]}`,
    distance,
  };
});

console.log(labeledRows);
