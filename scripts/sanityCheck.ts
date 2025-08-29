import { cos_sim } from "@huggingface/transformers";
import siglipTextEmbedder from "@/transformers/siglip-different-models/generateTextEmbedding";
import siglipImageEmbedder from "@/transformers/siglip-different-models/generateImageEmbedding";

type embeddingFunction = (arg: string) => Promise<number[]>;
interface CompareData {
  [model: string]: {
    similarity: number;
    dimension: number;
  };
}

const COMPARE_DATA: CompareData = {};

async function runCompare(
  model: string,
  generateTextEmbedding: embeddingFunction,
  generateImageEmbedding: embeddingFunction
) {
  const imageEmbedding = await generateImageEmbedding("./images/test.jpg");
  const textEmbedding = await generateTextEmbedding("a cat");

  if (imageEmbedding.length !== textEmbedding.length) {
    console.log("!!! ", model, "HAS MISMATCHED EMBEDDINGS");
  }

  COMPARE_DATA[model] = {
    similarity: cos_sim(textEmbedding, imageEmbedding),
    dimension: imageEmbedding.length,
  };
}

await Promise.allSettled([
  runCompare(
    "siglip-different-models",
    siglipTextEmbedder,
    siglipImageEmbedder
  ),
]);

console.table(COMPARE_DATA);
