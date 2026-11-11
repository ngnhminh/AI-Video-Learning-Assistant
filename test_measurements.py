import copy
import unittest
from summarize_evidence import calculate_call, summarize_calls

def sample():
    return {"call_id":1,"status":"ok","model_requested":"gemini-2.5-flash","latency_s":1,
            "usage":{"prompt_token_count":9462,"candidates_token_count":199,"thoughts_token_count":420,
                     "total_token_count":10081,"prompt_tokens_details":[{"modality":"TEXT","token_count":348},
                     {"modality":"VIDEO","token_count":8153},{"modality":"AUDIO","token_count":961}]}}

class MeasurementTests(unittest.TestCase):
    def test_live_sdk_modality_enum(self):
        from google.genai.types import MediaModality
        call=sample()
        for detail in call['usage']['prompt_tokens_details']:
            detail['modality']=MediaModality(detail['modality'])
        result=calculate_call(call)
        self.assertNotIn('MediaModality',result['cost_formula'])
        self.assertAlmostEqual(result['estimated_standard_usd'],.0050588)

    def test_multimodal_cost_includes_thinking(self):
        result=calculate_call(sample())
        self.assertAlmostEqual(result["estimated_standard_usd"],.0050588,12)
        self.assertIn("420 × 2.5 [THINKING]",result["cost_formula"])

    def test_cache_adjustment(self):
        call=sample()
        call["usage"]={"prompt_token_count":1000,"candidates_token_count":100,"thoughts_token_count":20,
            "prompt_tokens_details":[{"modality":"TEXT","token_count":1000}],
            "cached_content_token_count":400,"cache_tokens_details":[{"modality":"TEXT","token_count":400}]}
        self.assertAlmostEqual(calculate_call(call)["estimated_standard_usd"],.000492,12)
        del call["usage"]["cache_tokens_details"]
        self.assertIsNone(calculate_call(call)["estimated_standard_usd"])

    def test_absent_output_is_not_assumed_zero(self):
        call=sample();del call["usage"]["candidates_token_count"]
        self.assertIsNone(calculate_call(call)["estimated_standard_usd"])
        call["usage"]["total_token_count"]=9462
        call["usage"].pop("thoughts_token_count")
        self.assertEqual(calculate_call(call)["output_tokens_for_cost"],0)

    def test_partial_run_has_unknown_total(self):
        good=sample();before=copy.deepcopy(good)
        failed={"call_id":2,"status":"error","model_requested":"gemini-2.5-flash","latency_s":.2}
        result=summarize_calls([good,failed])
        self.assertIsNone(result["estimated_standard_usd"])
        self.assertIsNone(result["total_prompt_tokens"])
        self.assertAlmostEqual(result["known_cost_subtotal_usd"],.0050588)
        self.assertEqual(good,before)

    def test_unknown_variant_does_not_use_flash_price(self):
        call=sample();call["model_requested"]="gemini-2.5-flash-image"
        self.assertIsNone(calculate_call(call)["estimated_standard_usd"])

if __name__=="__main__":unittest.main()
