"""Protocol-critical offline checks; never call real providers."""
import importlib.util
from pathlib import Path
import tempfile
from threading import BoundedSemaphore, Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('omni_pilot',Path(__file__).with_name('run.py'))
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class FakeResponse:
    def __init__(self,text,finish='stop'):
        self.choices=[SimpleNamespace(finish_reason=finish,message=SimpleNamespace(content=text))]
        self.text,self.finish=text,finish
    def model_dump(self):
        return {'model':'offline-fake','usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30},
          'choices':[{'finish_reason':self.finish,'message':{'content':self.text}}]}


class FakeClient:
    def __init__(self):
        self.chat=SimpleNamespace(completions=self)
    def create(self,**kwargs):
        prompt=kwargs['messages'][0]['content']
        stage=pilot.classify(prompt)
        text={'reflect':'1. Evidence supports a conditional mechanism.\n2. Context changes applicability.',
              'expand':'-[Evidence]\n--{evidence query}\n-[Application]\n--{application query}',
              'outline_draft':'# Evidence\n## Mechanism\n# Application',
              'outline_refine':'# Evidence\n## Mechanism\n# Application',
              'write_section':'# Evidence\nThe observed mechanism holds under the stated conditions [1].\n\n# Application\nThe evidence informs the application [1].',
              'article_polish':'# Evidence\nThe observed mechanism holds under the stated conditions [1].\n\n# Application\nThe evidence informs the application [1].'}[stage]
        return FakeResponse(text)


class FakeSearch:
    def __init__(self,key,audit):self.audit=audit
    def __call__(self,queries,exclude_urls=None):
        queries=[queries] if isinstance(queries,str) else queries
        data=[{'url':f'https://example.org/{n}','title':f'Paper {n}','snippets':[f'Evidence {n} supports a conditional mechanism.']} for n in (1,2)]
        for q in queries:
            self.audit.record('search_results.jsonl',{'query':q,'results':data})
            self.audit.record('search_usage.jsonl',{'status_code':200})
        return data


class FakeEncoder:
    def encode(self,things,**kwargs):
        import numpy as np
        if isinstance(things,str):return np.array([1.,0.,0.])
        return np.array([[1.,0.,0.] for x in things])


class Checks(unittest.TestCase):
    def test_section_uses_all_sources_once(self):
        from src.actions.article_generation import ConvToSection
        writer=ConvToSection('WriteSection',FakeClient())
        calls=[]
        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output='# Test\nSupported finding [1].')
        writer.write_section=generate
        writer.forward('Topic','','Test',[{'snippets':['SOURCE_A']},{'snippets':['SOURCE_B']}],'')
        self.assertEqual(len(calls),1)
        self.assertIn('SOURCE_A',calls[0]['info']);self.assertIn('SOURCE_B',calls[0]['info'])

    def test_invalid_citation_keeps_last_valid_source(self):
        article=pilot.Article('Topic')
        article.update_section('# Evidence\nSupported [2], unsupported [3].',
            [{'url':'https://example.org/1','snippets':['a']},{'url':'https://example.org/2','snippets':['b']}])
        self.assertIn('https://example.org/2',article.reference['url_to_unified_index'])
        self.assertNotIn('[3]',article.to_string())

    def test_truncated_response_retains_usage_and_fails(self):
        with tempfile.TemporaryDirectory() as d:
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:FakeResponse('partial','length'))))
            audit=pilot.Audit(Path(d),Event(),BoundedSemaphore(1))
            lm=pilot.AuditedLM(client,audit)
            with self.assertRaises(pilot.IncompleteGeneration):lm('Test')
            self.assertEqual(pilot.summarize_usage(Path(d))['total_tokens'],30)

    def test_full_pipeline_offline(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pilot,'AuditedSearch',FakeSearch):
            row=pilot.run_task(1,{'topic':'Offline topic','intent':'Understand the mechanism','domain':'test'},
                Path(d),FakeEncoder(),FakeClient(),Event(),BoundedSemaphore(2),('not-a-key','not-a-key'))
            if row['status']!='ok':
                print((Path(row['directory'])/'diagnostic.txt').read_text())
            self.assertEqual(row['status'],'ok')
            self.assertEqual(row['tree']['depth'],2)
            self.assertEqual(row['tree']['nodes'],3)
            self.assertEqual(row['usage']['unaccounted_model_requests'],0)
            self.assertTrue((Path(row['directory'])/'article_with_sources.md').exists())

    def test_extension_failure_is_not_suppressed(self):
        from src.tools.mindmap import MindPoint
        with patch.object(MindPoint,'extend',side_effect=RuntimeError('offline extension failure')):
            model=pilot.LocalMindMap(retriever=lambda q:[{'url':'https://example.org','snippets':['evidence']}],
                gen_concept_lm=None,depth=2,local_encoder=FakeEncoder())
            model.concept_generator=lambda x:['concept']
            with self.assertRaisesRegex(RuntimeError,'offline extension failure'):
                list(model.build_map('Topic'))


if __name__=='__main__':unittest.main(verbosity=2)
