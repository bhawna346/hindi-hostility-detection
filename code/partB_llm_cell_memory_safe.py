import os, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
LLMS = {'Qwen2.5-1.5B-Instruct':'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen2.5-7B-Instruct':'Qwen/Qwen2.5-7B-Instruct'}
SYS = ('You are a content moderator for Hindi social media posts. Classify the post. '
       'If it is not hostile, answer exactly: non-hostile. Otherwise answer with one or more of '
       'fake, hate, offensive, defamation separated by commas. '
       'fake = false or misleading news; hate = attacks a group by religion, caste, gender etc.; '
       'offensive = abusive, insulting or vulgar language; defamation = false claims damaging a specific person or organisation. '
       'Answer with the labels only.')
rng = np.random.default_rng(SEED); shots = []
for lab in ['non-hostile','fake','hate','offensive','defamation']:
    idx = np.where(tr['Labels Set'].str.strip() == lab)[0]; shots.append(int(rng.choice(idx)))

def parse(txt):
    t = txt.lower(); f = [int(l in t) for l in FINE]
    return int(any(f)), f

def run_llm(path, k_shot, T, variant):
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = 'left'
    m = AutoModelForCausalLM.from_pretrained(path, device_map='auto',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))
    def msgs(post):
        ms = [{'role':'system','content':SYS}]
        if k_shot:
            for s in shots:
                ex = clean(tr['Post'][s]) if variant == 'clean' else tr['Post'][s]
                ms += [{'role':'user','content':ex[:600]}, {'role':'assistant','content':tr['Labels Set'][s]}]
        return ms + [{'role':'user','content':post[:600]}]
    prompts = [tok.apply_chat_template(msgs(p), tokenize=False, add_generation_prompt=True) for p in T['te']]
    bs = 16 if k_shot == 0 else 4          # 5-shot prompts are long, so smaller batches
    outs = []; t0 = time.perf_counter(); b = 0
    while b < len(prompts):
        try:
            e = tok(prompts[b:b+bs], return_tensors='pt', padding=True).to('cuda')
            with torch.no_grad():
                g = m.generate(**e, max_new_tokens=15, do_sample=False)
            outs += tok.batch_decode(g[:, e['input_ids'].shape[1]:], skip_special_tokens=True); b += bs
        except torch.cuda.OutOfMemoryError:
            e = g = None; gc.collect(); torch.cuda.empty_cache(); bs = max(1, bs // 2); print('  out of memory, batch size now', bs)
    inf = (time.perf_counter() - t0) / len(prompts) * 1e6
    del m; gc.collect(); torch.cuda.empty_cache()
    parsed = [parse(o) for o in outs]
    return np.array([p[0] for p in parsed]), np.array([p[1] for p in parsed]), inf, outs

done = pd.read_csv('llm_results.csv') if os.path.exists('llm_results.csv') else pd.DataFrame()
rows = done.to_dict('records')
for variant in ['raw', 'clean']:
  for name, path in LLMS.items():
    for k in (0, 5):
        if not done.empty and ((done.model == name) & (done.text == variant) & (done.setting == f'{k}-shot')).any():
            print('skip (already done):', name, variant, f'{k}-shot'); continue
        print('running:', name, variant, f'{k}-shot')
        pc, pf, inf, raw = run_llm(path, k, TEXT[variant], variant)
        r = dict(model=name, text=variant, setting=f'{k}-shot', infer_ms_per_1k_GPU=inf, **score(pc, pf)); rows.append(r); print(r)
        json.dump(raw, open(f'raw_{name}_{k}shot_{variant}.json','w'), ensure_ascii=False)
        pd.DataFrame(rows).to_csv('llm_results.csv', index=False)
pd.DataFrame(rows)
