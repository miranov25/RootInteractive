"""Canonical runner packet and truthful failure-status integration checks."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

class ReviewRunnerContracts(unittest.TestCase):
    def test_current_worktree_sources_new_files_and_collection_failure(self):
        source=Path(__file__).resolve().parents[3]/'run_tests.sh'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); shutil.copy2(source,root/'run_tests.sh')
            dirs=['RootInteractive/InteractiveDrawing/bokeh','RootInteractive/Tools','RootInteractive/onnx','RootInteractive/MLpipeline/tests','RootInteractive/InteractiveDrawing/bokeh/headless']  # headless: own entry since PHASE_4 stage 4.2c-2b (last, so dirs[1] stays Tools)
            for directory in dirs:
                p=root/directory;p.mkdir(parents=True);(p/'test_smoke.py').write_text('def test_ok(): assert True\n')
            example=root/'RootInteractive/MLpipeline/examples/phase_2_1_reducible_example.py';example.parent.mkdir();example.write_text("print('example PASS')\n")
            def git(*args):
                subprocess.run(['git','-C',d,*args],check=True,capture_output=True)
            git('init');git('config','user.email','fixture@example.invalid');git('config','user.name','Runner fixture')
            git('add','--','run_tests.sh',*[directory+'/test_smoke.py' for directory in dirs],str(example.relative_to(root)))
            git('commit','-m','fixture baseline');git('tag','PHASE_BEGIN_RootInteractive')
            added=root/'RootInteractive/MLpipeline/tests/test_new.py';added.write_text('def test_new(): assert True\n')
            tracked=root/dirs[1]/'test_smoke.py';tracked.write_text('def test_ok(): assert 1 == 1\n')
            env=dict(os.environ);env['RootInteractive_DOCS']=str(root/'docs')
            run=subprocess.run(['./run_tests.sh','--quick','-n','1'],cwd=d,env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            archive=root/'test_logs/reviewer_latest.zip'
            self.assertTrue((root/'docs/files_RI'/archive.resolve().name).is_file())
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('source/RootInteractive/MLpipeline/tests/test_new.py'),added.read_bytes())
                self.assertIn(b'test_new.py',z.read('diff_untracked.patch'))
                self.assertIn(b'assert 1 == 1',z.read('diff_HEAD_to_worktree.patch'))
                self.assertIn('tests_report.json',z.namelist())
                self.assertIn('environment.json',z.namelist())
                self.assertIn('MLpipeline_example.log',z.namelist())
            tracked.write_text('this is broken syntax !!!\n')
            failed=subprocess.run(['./run_tests.sh','--quick','-n','1'],cwd=d,env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0,failed.stdout+failed.stderr)
            with zipfile.ZipFile(archive) as z:
                self.assertIn(b'SyntaxError',z.read('Tools_test.log'))
                self.assertEqual(z.read('source/RootInteractive/Tools/test_smoke.py'),tracked.read_bytes())


    def test_matrix_failures_stale_outputs_and_missing_phase_tag(self):
        source = Path(__file__).resolve().parents[3] / 'run_tests.sh'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copy2(source, root / 'run_tests.sh')
            def git(*args):
                subprocess.run(['git', '-C', d, *args], check=True, capture_output=True)
            git('init'); git('config', 'user.email', 'fixture@example.invalid')
            git('config', 'user.name', 'Runner fixture')
            git('add', '--', 'run_tests.sh'); git('commit', '-m', 'fixture')
            git('tag', 'PHASE_BEGIN_RootInteractive')
            generator = root / 'scripts/generate_capability_matrix.py'
            generator.parent.mkdir()
            matrix = root / 'RootInteractive/InteractiveDrawing/bokeh/doc/CAPABILITY_MATRIX.md'
            matrix.parent.mkdir(parents=True)
            env = dict(os.environ, RootInteractive_DOCS=str(root / 'docs'))
            def run(*args):
                return subprocess.run(['./run_tests.sh', *args], cwd=d, env=env,
                                      capture_output=True, text=True)
            for with_reports in (False, True):
                if with_reports:
                    logs = root / 'test_logs/bokeh_fixture'
                    logs.mkdir(parents=True)
                    (logs / 'report.json').write_text('{}')
                    (logs.parent / 'bokeh_latest').symlink_to('bokeh_fixture')
                for body in ('raise RuntimeError("injected generator failure")', 'pass'):
                    generator.write_text(body + '\n')
                    matrix.write_text('STALE')
                    matrix.with_suffix('.json').write_text('{}')
                    result = run('--matrix', '-n', '1')
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertFalse(matrix.exists())
            generator.write_text("from pathlib import Path\np=Path('RootInteractive/InteractiveDrawing/bokeh/doc/CAPABILITY_MATRIX.md')\np.write_text('fresh')\np.with_suffix('.json').write_text('{}')\n")
            result = run('--matrix', '-n', '1')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            git('tag', '-d', 'PHASE_BEGIN_RootInteractive')
            result = run('--matrix', '-n', '1')
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('phase review blocked (P0)', result.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
