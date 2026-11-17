"""Local estimated API spending limits shared by grasp/place instances in a process."""
import json
import threading
import time
from pathlib import Path


class VisionBudgetError(RuntimeError):
    api_status, retry_after_s, fatal = 'budget', None, True


class SpendLedger:
    # USD per million input, cached input, output tokens; verified 10 Nov 2026.
    RATES = {'gpt-6.1-terra': (1., .05, 5.), 'gpt-6.1-sol': (2., .1, 10.), 'gpt-6-sol': (2., .2, 10.),
             'gpt-6-astra': (10., 1., 50.), 'gpt-6-luna': (.1, .01, .5)}

    def __init__(self, path=None, session_limit=5., move_limit=.50, move_calls=40):   # 2026-11-10: Terra looks cost ~$0.0025; the
        # mid-descent and fresh looks plus retries exceeded 12 calls per move and the guard failed moves with $9 left on the account
        self.path = Path(path) if path else None
        self.session_limit, self.move_limit, self.move_calls = session_limit, move_limit, move_calls
        self.spent = self.reserved = 0.
        self.lock = threading.Lock()

    def rates(self, model, tier):
        rates = self.RATES.get(model, self.RATES['gpt-6-astra'])
        factor = {'fast':2., 'priority':2., 'ultrafast':6., 'flex':.5}.get(tier, 1.)
        return [rate*factor for rate in rates]

    def reserve(self, owner, model, tier, output_limit):
        rates = self.rates(model,tier)
        estimate = (8192*rates[0]+output_limit*rates[2])/1e6
        with self.lock:
            state = getattr(owner, '_vision_spend', None)
            if state is None:
                state = dict(spent=0.,reserved=0.,calls=0)
                owner._vision_spend = state
            if (self.spent+self.reserved+estimate > self.session_limit or
                    state['spent']+state['reserved']+estimate > self.move_limit or
                    state['calls'] >= self.move_calls):
                raise VisionBudgetError('Local estimated API budget reached; holding the arm')
            state['calls'] += 1; state['reserved'] += estimate; self.reserved += estimate
            return state, estimate, rates, model, tier

    def finish(self, ticket, usage=None, known_unbilled=False):
        state, estimate, rates, model, tier = ticket
        usage = usage or {}
        i = usage.get('input_tokens',0); o = usage.get('output_tokens',0)
        cached = (usage.get('input_tokens_details') or {}).get('cached_tokens',0)
        cost = 0. if known_unbilled else ((max(0,i-cached)*rates[0]+cached*rates[1]+o*rates[2])/1e6 if usage else estimate)
        with self.lock:
            self.reserved -= estimate; state['reserved'] -= estimate
            self.spent += cost; state['spent'] += cost
            row = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),model=model,tier=tier,input_tokens=i,
                       output_tokens=o,cached_tokens=cached,estimated_usd=round(cost,6),
                       move_estimated_usd=round(state['spent'],6),session_estimated_usd=round(self.spent,6))
            if self.path:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with self.path.open('a') as f: f.write(json.dumps(row)+'\n')
            return row


SESSION_LEDGER = SpendLedger(Path(__file__).resolve().parent.parent/'data/logs/astra_api_usage.jsonl')
