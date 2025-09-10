import {
  AutoTokenizer,
  AutoProcessor,
  SiglipModel,
  RawImage,
  Tensor,
} from "@huggingface/transformers";
import { createEmptyTensor } from "../transformers/utils";

// Load tokenizer, processor, and model
const tokenizer = await AutoTokenizer.from_pretrained(
  "Xenova/siglip-base-patch16-224"
);
const processor = await AutoProcessor.from_pretrained(
  "Xenova/siglip-base-patch16-224"
);
const model = await SiglipModel.from_pretrained(
  "Xenova/siglip-base-patch16-224"
);

// Run tokenization
const texts = ["a photo of 2 cats", "a photo of 2 dogs"];
const text_inputs = tokenizer(texts, {
  padding: "max_length",
  truncation: true,
});

// Read image and run processor
const image = await RawImage.read(
  "http://images.cocodataset.org/val2017/000000039769.jpg"
);
const image_inputs = await processor(image);

// create correctly sized empty tensor

const emptyImageInputs = {
  pixel_values: createEmptyTensor("float32", [1, 3, 224, 224]),
};

const emptyTextInputs = {
  input_ids: createEmptyTensor("int64", [1, 1]),
};

// const output = await model({ ...text_inputs, ...image_inputs });
// const output = await model({ ...text_inputs, ...emptyImageInputs });
const output = await model({ ...emptyTextInputs, ...image_inputs });
console.log(output);
debugger;
