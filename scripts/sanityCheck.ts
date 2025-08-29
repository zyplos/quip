import { cos_sim } from "@huggingface/transformers";
import generateTextEmbedding from "@/transformers/siglip-different-models/generateTextEmbedding";
import generateImageEmbedding from "@/transformers/siglip-different-models/generateImageEmbedding";

const imageEmbedding = await generateImageEmbedding("./images/test.jpg");
console.log(imageEmbedding.length);

const textEmbedding = await generateTextEmbedding("a cat");
console.log(textEmbedding.length);

console.log(
  "same embedding length",
  imageEmbedding.length === textEmbedding.length
);

console.log(cos_sim(textEmbedding, imageEmbedding));
