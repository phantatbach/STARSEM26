import pandas as pd
import json
# from frame_semantic_transformer import FrameSemanticTransformer
import os
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# frame_transformer_base = FrameSemanticTransformer('base', batch_size=30)
# frame_transformer_small = FrameSemanticTransformer('small', batch_size=8)

selected_lemmas = ['attack_nn', 'bag_nn', 'ball_nn', 'bit_nn', 'chairman_nn', 'circle_vb', 'contemplation_nn', 'donkey_nn', 'edge_nn', 'face_nn', 'fiction_nn', 'gas_nn', 'graft_nn', 'head_nn', 'land_nn', 'lane_nn', 'lass_nn', 'multitude_nn', 'ounce_nn', 'part_nn', 'pin_vb', 'plane_nn', 'player_nn', 'prop_nn', 'quilt_nn', 'rag_nn', 'record_nn', 'relationship_nn', 'risk_nn', 'savage_nn', 'stab_nn', 'stroke_vb', 'thump_nn', 'tip_vb', 'tree_nn', 'twist_nn', 'word_nn']
modes = ['lemma', 'token']

# Helper function to convert detection results to a serializable record
def detect_result_to_record(result, meta=None):
    if meta is None:
        meta = {}

    record = {
        **meta,
        "sentence": getattr(result, "sentence", None),
        "trigger_locations": getattr(result, "trigger_locations", None),
        "frames": []
    }

    frames = getattr(result, "frames", []) or []
    for fr in frames:
        fr_dict = {
            "name": getattr(fr, "name", None),
            "trigger_location": getattr(fr, "trigger_location", None),
            "frame_elements": []
        }

        fes = getattr(fr, "frame_elements", []) or []
        for fe in fes:
            fr_dict["frame_elements"].append({
                "name": getattr(fe, "name", None),
                "text": getattr(fe, "text", None)
            })

        record["frames"].append(fr_dict)

    return record

def save_total_results_jsonl(total_results, output_path, lemma=None, mode=None):
    with open(output_path, "w", encoding="utf-8") as f:
        for corpus_name, results in total_results.items():
            for result in results:
                meta = {
                    "lemma": lemma,
                    "mode": mode,
                    "corpus": corpus_name,
                }
                rec = detect_result_to_record(result, meta=meta)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def worker(gpu_id: int, lemmas_subset: list[str], mode: str):
    # Pin THIS process to one GPU (visible as cuda:0 inside the process)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import after setting CUDA_VISIBLE_DEVICES
    import pandas as pd
    from tqdm import tqdm
    from frame_semantic_transformer import FrameSemanticTransformer

    frame_transformer_base = FrameSemanticTransformer("base", batch_size=24, use_gpu=True)
    frame_transformer_small = FrameSemanticTransformer("small", batch_size=8, use_gpu=True)

    # Optional: load weights immediately once (models are lazy by default)
    frame_transformer_base.setup()
    frame_transformer_small.setup()

    output_dir = f"... /FrameNet_parsed/{mode}" # <-- Adjust path as needed 
    os.makedirs(output_dir, exist_ok=True)

    for selected_lemma in tqdm(lemmas_subset, desc=f"GPU{gpu_id} {mode}"):
        corpus_1_path = f"... /corpus1/{mode}/ccoha1_{selected_lemma}.csv" # <-- Adjust path as needed
        df1 = pd.read_csv(corpus_1_path, dtype=str, keep_default_na=False)
        corpus_1 = df1["sent"].astype(str).str.strip()
        corpus_1 = corpus_1[corpus_1 != ""].tolist()

        corpus_2_path = f"... /corpus2/{mode}/ccoha2_{selected_lemma}.csv" # <-- Adjust path as needed
        df2 = pd.read_csv(corpus_2_path, dtype=str, keep_default_na=False)
        corpus_2 = df2["sent"].astype(str).str.strip()
        corpus_2 = corpus_2[corpus_2 != ""].tolist()

        corpora = {"corpus_1": corpus_1, "corpus_2": corpus_2}
        total_results = {name: [] for name in corpora}

        for name, sentences in corpora.items():
            base_results = frame_transformer_base.detect_frames_bulk(sentences)

            for base_result in base_results:
                org_sent = base_result.sentence

                if not base_result.frames:
                    print(f"No frames detected in base model for sentence: {org_sent} in mode {mode}. Running small model...")
                    small_result = frame_transformer_small.detect_frames(org_sent)
                    total_results[name].append(small_result if small_result.frames else small_result)
                else:
                    total_results[name].append(base_result)

        output_path = f"{output_dir}/{selected_lemma}_{mode}_FrameNet_parsed.jsonl"
        save_total_results_jsonl(total_results, output_path, lemma=selected_lemma, mode=mode)

def main():
    selected_lemmas = [
        "attack_nn","bag_nn","ball_nn","bit_nn","chairman_nn","circle_vb","contemplation_nn","donkey_nn",
        "edge_nn","face_nn","fiction_nn","gas_nn","graft_nn","head_nn","land_nn","lane_nn","lass_nn",
        "multitude_nn","ounce_nn","part_nn","pin_vb","plane_nn","player_nn","prop_nn","quilt_nn","rag_nn",
        "record_nn","relationship_nn","risk_nn","savage_nn","stab_nn","stroke_vb","thump_nn","tip_vb",
        "tree_nn","twist_nn","word_nn"
    ]
    modes = ["lemma", "token"]

    # Split lemmas across the 2 GPUs (simple even/odd split)
    lemmas_gpu0 = selected_lemmas[::2]
    lemmas_gpu1 = selected_lemmas[1::2]

    for mode in modes:
        with ProcessPoolExecutor(max_workers=2) as ex:
            ex.submit(worker, 0, lemmas_gpu0, mode)
            ex.submit(worker, 2, lemmas_gpu1, mode)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) 
    main()