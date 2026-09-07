import os
import json
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI


PROMPT = "Describe the image in 3-5 sentences!"

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def atomic_write_json(path, data):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_existing_dataset(output_path):
    if not os.path.exists(output_path):
        return [], set()

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    completed_ids = {item["id"] for item in data}
    return data, completed_ids


def main(args):
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )

    image_files = sorted([
        f for f in os.listdir(args.image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    # Resume support
    dataset, completed_ids = load_existing_dataset(args.output)

    print(f"🔁 Resuming: {len(completed_ids)} images already processed")

    for image_name in tqdm(image_files, desc="Captioning images"):
        image_id = os.path.splitext(image_name)[0]

        if image_id in completed_ids:
            continue

        image_path = os.path.join(args.image_folder, image_name)
        image_base64 = encode_image(image_path)
        mime_type = MIME_TYPES.get(os.path.splitext(image_name)[1].lower(), "image/jpeg")

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2
            )

            caption = response.choices[0].message.content.strip()

            dataset.append({
                "id": image_id,
                "image": image_name,
                "caption": caption
            })

            completed_ids.add(image_id)

            # Incremental save (crash-safe: write to temp file, then atomic replace)
            atomic_write_json(args.output, dataset)

        except Exception as e:
            print(f"⚠️ Failed on {image_name}: {e}")

    print(f"\n✅ Finished. Total captions: {len(dataset)}")
    print(f"✅ Output saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create image caption dataset with resume support"
    )

    parser.add_argument("--image-folder", required=True, help="Path to image folder")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--api-base", required=True, help="OpenAI API base URL")
    parser.add_argument("--model", default="gpt-4o-mini", help="Vision-capable model name")

    args = parser.parse_args()
    main(args)