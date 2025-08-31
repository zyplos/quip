import { cos_sim } from "@huggingface/transformers";
//
import siglipDifferentTextEmbedder from "@/transformers/siglip-different-models/generateTextEmbedding";
import siglipDifferentImageEmbedder from "@/transformers/siglip-different-models/generateImageEmbedding";
//
import clipDifferentTextEmbedder from "@/transformers/clip-different-models/generateTextEmbedding";
import clipDifferentImageEmbedder from "@/transformers/clip-different-models/generateImageEmbedding";
//
import clipProjectedTextEmbedder from "@/transformers/clip-models-with-projection/generateTextEmbedding";
import clipProjectedImageEmbedder from "@/transformers/clip-models-with-projection/generateImageEmbedding";
//
import siglipModelTextEmbedder from "@/transformers/siglip-model/generateTextEmbedding";
import siglipModelImageEmbedder from "@/transformers/siglip-model/generateImageEmbedding";

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
    siglipDifferentTextEmbedder,
    siglipDifferentImageEmbedder
  ),
  runCompare(
    "clip-different-models",
    clipDifferentTextEmbedder,
    clipDifferentImageEmbedder
  ),
  runCompare(
    "clip-models-with-projection",
    clipProjectedTextEmbedder,
    clipProjectedImageEmbedder
  ),
  runCompare("siglip-model", siglipModelTextEmbedder, siglipModelImageEmbedder),
]);

console.table(COMPARE_DATA);
