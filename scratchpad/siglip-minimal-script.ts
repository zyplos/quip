import {
  AutoProcessor,
  AutoTokenizer,
  cos_sim,
  matmul,
  RawImage,
  SiglipModel,
  type Tensor,
} from "@huggingface/transformers";

import { MODEL_CONFIG, SIGLIP_MODEL_NAME } from "@/transformers/config";
import { createEmptyTensor } from "@/transformers/utils";

const imageProcessor = await AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME);
const textProcessor = await AutoTokenizer.from_pretrained(SIGLIP_MODEL_NAME);
const model = await SiglipModel.from_pretrained(
  SIGLIP_MODEL_NAME,
  MODEL_CONFIG
);

const emptyTextInputs = {
  input_ids: createEmptyTensor("int64", [2, 64]),
};

async function generateImageEmbedding(path: string): Promise<Tensor> {
  const image = await RawImage.read(path);
  const imageInputs = await imageProcessor(image);
  const { image_embeds } = await model({ ...emptyTextInputs, ...imageInputs });

  return image_embeds;
}

const emptyImageInputs = {
  pixel_values: createEmptyTensor("float32", [1, 3, 224, 224]),
};

async function generateTextEmbedding(text: string): Promise<Tensor> {
  const textInputs = await textProcessor(text);
  const { text_embeds } = await model({ ...textInputs, ...emptyImageInputs });

  return text_embeds;
}

const imageEmb = await generateImageEmbedding(
  "C:/devspace/frontend/quip/images/test.jpg"
);

const textEmb = await generateTextEmbedding(
  "Represent the query for retrieving evidence documents: a cat"
);

console.log(imageEmb.size);
console.log(textEmb.size);

const scored = await matmul(textEmb, imageEmb.transpose(1, 0));

// console.log("COS", cos_sim(imageEmb.tolist()[0], textEmb.tolist()[0]));
console.log("MAT", scored.item());
