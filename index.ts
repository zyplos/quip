import { serve, type BunRequest } from "bun";
import homepage from "./src/index.html";
import loadImageEmbeddings from "./embeddingManager/loadImageEmbeddings";
import { cos_sim } from "@huggingface/transformers";
import type { QueryResponse } from "./Types";
import generateTextEmbedding from "./transformers/jina-v1-clip/generateTextEmbedding";

const imageEmbeddings = await loadImageEmbeddings();

const server = serve({
  routes: {
    "/": homepage,

    "/api/search": {
      async POST(req: BunRequest) {
        const { query } = await req.json();
        const textEmbedding = (await generateTextEmbedding(
          query
        )) as unknown as number[];

        const results = findSimilarImages(textEmbedding, imageEmbeddings);
        // console.log(results);

        return Response.json({ results } as QueryResponse);
      },
    },
  },

  development: true,

  async fetch(req) {
    const path = new URL(req.url).pathname;
    const file = Bun.file(`./${path}`);
    return new Response(file);
  },
});

console.log(`Listening on ${server.url}`);

function findSimilarImages(
  targetEmbedding: number[],
  allEmbeddings: Map<string, number[]>,
  topN = 100
): { filename: string; similarity: number }[] {
  const similarities: { filename: string; similarity: number }[] = [];

  // Calculate similarity for each image
  for (const [filename, embedding] of allEmbeddings.entries()) {
    const similarity = cos_sim(targetEmbedding, embedding);
    similarities.push({ filename, similarity });
  }

  // Sort by similarity in descending order
  similarities.sort((a, b) => b.similarity - a.similarity);

  // Return the top N results
  return similarities.slice(0, topN);
}
