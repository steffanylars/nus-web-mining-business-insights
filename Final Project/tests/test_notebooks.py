"""Checks on the step notebooks: valid files, no long dashes, and code that matches the scripts.

The notebooks explain the pipeline cell by cell; their code is cut from src/*.py. These tests fail if
a script is edited without regenerating the matching notebook, so the two cannot drift apart.
"""
import ast
import json
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'
SRC = ROOT / 'src'
FORBIDDEN = '\u2014\u2013\u2212'  # em dash, en dash, minus sign

STEPS = {
    'Step2_Build_Dataset.ipynb': 'step2_build_dataset.py',
    'Step3_Feature_Engineering Final.ipynb': 'step3_features.py',
    'Step4_Baselines_LambdaMART.ipynb': 'step4_baselines_lgbm.py',
    'Step5_Neural_Rankers_DHAR.ipynb': 'step5_neural.py',
    'Step6_Diagnostics.ipynb': 'step6_diagnostics.py',
    'Step7_ColdStart_Diversity.ipynb': 'step7_coldstart_diversity.py',
}
# Lines of main() that the notebook rewrites on purpose (command-line parsing, shared setup).
REWRITTEN = ('args', 'parser', 'prepare()', 'sys.argv')


def code_of(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    return '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')


class StepNotebooks(unittest.TestCase):
    def test_all_steps_present(self):
        names = {p.name for p in NOTEBOOKS.glob('Step*.ipynb')}
        self.assertEqual(names, set(STEPS) | {'Step1_Temporal_Analysis.ipynb'})

    def test_valid_notebooks_without_long_dashes(self):
        for path in sorted(NOTEBOOKS.glob('Step*.ipynb')):
            text = path.read_text(encoding='utf-8')
            nbformat.validate(nbformat.reads(text, as_version=4))
            found = {hex(ord(c)): text.count(c) for c in FORBIDDEN if c in text}
            self.assertEqual(found, {}, f'{path.name} contains long dashes: {found}')

    def test_every_function_of_the_script_that_the_notebook_shows_is_verbatim(self):
        for notebook, script in STEPS.items():
            text = (SRC / script).read_text(encoding='utf-8')
            lines = text.splitlines()
            code = code_of(NOTEBOOKS / notebook)
            shown = 0
            functions = [n for n in ast.parse(text).body
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != 'main']
            for node in functions:
                segment = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                if segment in code:
                    shown += 1
                else:
                    # a function may only be imported instead of shown (Step 7 imports fit_ranker)
                    self.assertTrue(f'import {node.name}' in code, f'{notebook}: {node.name} neither shown nor imported')
            if functions:  # Step 3 has only main(), whose body is checked by the next test
                self.assertGreater(shown, 0, f'{notebook} shows none of the functions of {script}')

    def test_main_body_of_the_script_appears_in_the_notebook(self):
        for notebook, script in STEPS.items():
            if notebook.startswith('Step7'):
                continue  # Step 7 splits main() into hand-written cells that call the same functions
            text = (SRC / script).read_text(encoding='utf-8')
            tree = ast.parse(text)
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            lines = text.splitlines()[main.body[0].lineno - 1:main.end_lineno]
            code_lines = {l.strip() for l in code_of(NOTEBOOKS / notebook).splitlines()}
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#') or any(token in stripped for token in REWRITTEN):
                    continue
                if stripped == "if __name__ == '__main__':":
                    continue
                self.assertTrue(stripped in code_lines, f'{notebook}: line of {script} main() missing: {stripped}')


if __name__ == '__main__':
    unittest.main()
