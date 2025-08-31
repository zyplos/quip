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
//
import jinaV2AutoTextEmbedder from "@/transformers/jina-v2-auto/generateTextEmbedding";
import jinaV2AutoImageEmbedder from "@/transformers/jina-v2-auto/generateImageEmbedding";
//
import jinaV1CLIPTextEmbedder from "@/transformers/jina-v1-clip/generateTextEmbedding";
import jinaV1CLIPImageEmbedder from "@/transformers/jina-v1-clip/generateImageEmbedding";
import { MODEL_CONFIG } from "@/transformers/config";
import { markdownTable, type CompareData } from "@/transformers/utils";

type embeddingFunction = (arg: string) => Promise<number[]>;

const QUERY = "a cat";
// const QUERY =
// "Represent the query for retrieving evidence documents: an image of a cat";
// const QUERY = "Represent the query for retrieving images: a cat";

const IMAGE_PATH = "./images/other-test.jpg";
const COMPARE_DATA: CompareData = {};

async function runCompare(
  model: string,
  generateTextEmbedding: embeddingFunction,
  generateImageEmbedding: embeddingFunction
) {
  const textEmbedding = await generateTextEmbedding(QUERY);
  const imageEmbedding = await generateImageEmbedding(IMAGE_PATH);

  if (textEmbedding.length !== imageEmbedding.length) {
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
  runCompare("jina-v2-auto", jinaV2AutoTextEmbedder, jinaV2AutoImageEmbedder),
  runCompare("jina-v1-clip", jinaV1CLIPTextEmbedder, jinaV1CLIPImageEmbedder),
]);

console.log(`device ${MODEL_CONFIG.device}`);
console.log(`"${QUERY}"`);
console.log(markdownTable(COMPARE_DATA));
