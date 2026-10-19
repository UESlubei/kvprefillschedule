"""Keep the strict failure and assess native-repeat + semantic evidence separately."""
import json
import math
from pathlib import Path
from latest_v0320.common import DATA,ROOT,dump

def main():
    report=ROOT/'reports/latest_v0320'
    strict=json.loads((report/'teacher-forced-audit.json').read_text())
    cases=[]
    for case in strict['cases']:
        folder=Path(case['evidence']);spec=json.loads((folder/'spec.json').read_text())
        candidate=spec['args']['policy']
        ref={r['key']:r for r in json.loads((folder/'N0.json').read_text())['rows']}[case['key']]
        opt={r['key']:r for r in json.loads((folder/f'{candidate}.json').read_text())['rows']}[case['key']]
        tokens=list(map(str,[case['reference_token'],case['candidate_token']]))
        pdelta=max(abs(math.exp(ref['logprobs'][t])-math.exp(opt['logprobs'][t])) for t in tokens)
        native=[ref]
        for file in sorted(folder.glob('N0-repeat-*.json')):
            rows={r['key']:r for r in json.loads(file.read_text())['rows']}
            if case['key'] in rows:native.append(rows[case['key']])
        native_top1=sorted({r['predicted_token'] for r in native})
        native_range=max(max(math.exp(r['logprobs'][t]) for r in native)-
            min(math.exp(r['logprobs'][t]) for r in native) for t in tokens)
        top1_supported=case['same_history_top1_equal'] or (len(native)>=3 and opt['predicted_token'] in native_top1)
        cases.append({'key':case['key'],'max_probability_delta':pdelta,
            'native_replicas':len(native),'native_top1_set':native_top1,
            'native_probability_range':native_range,'native_supported_top1':top1_supported,
            'supported':pdelta<=.001 and case['near_tie_0_05'] and top1_supported})
    semantic=[]
    for track in ['eager','default']:
        for policy in ['N0','NT']:
            file=DATA/f'raw/semantic/{track}-{policy}.json'
            data=json.loads(file.read_text())
            assert len(data['rows'])==9
            semantic.append({'track':track,'policy':policy,'correct':sum(r['correct'] for r in data['rows']),
                'total':len(data['rows']),'exact_across_cache_modes':data['exact_across_cache_modes'],
                'passed':data['passed'],'evidence':str(file)})
    result={'original_strict_logprob_check_passed':strict['passed'],
        'additional_diagnostic_criteria':{'max_absolute_probability_delta':.001,
            'near_tie_logprob_gap':.05,'changed_teacher_top1_requires_native_self_repeat_support':True,
            'status':'additional explanatory checks after pilot; not a preregistered accuracy guarantee'},
        'cases':cases,'semantic_checks':semantic,
        'supported_for_performance_experiment':all(r['supported'] for r in cases) and all(s['passed'] for s in semantic),
        'scope':'finite probability and semantic tests; original strict failure is retained, no arbitrary-batch bitwise equivalence claim'}
    dump(report/'quality-followup-assessment.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['supported_for_performance_experiment']:raise SystemExit(2)

if __name__=='__main__':main()
