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
            root=Path(d); shutil.copyfile(source,root/'run_tests.sh')
            dirs=['RootInteractive/InteractiveDrawing/bokeh','RootInteractive/Tools','RootInteractive/onnx','RootInteractive/MLpipeline/tests']
            for directory in dirs:
                p=root/directory;p.mkdir(parents=True);(p/'test_smoke.py').write_text('def test_ok(): assert True\n')
            example=root/'RootInteractive/MLpipeline/examples/phase_2_1_reducible_example.py';example.parent.mkdir();example.write_text("print('example PASS')\n")
            def git(*args):
                subprocess.run(['git','-C',d,*args],check=True,capture_output=True)
            git('init');git('config','user.email','fixture@example.invalid');git('config','user.name','Runner fixture')
            git('add','--','run_tests.sh',*[directory+'/test_smoke.py' for directory in dirs],str(example.relative_to(root)))
            git('commit','-m','fixture baseline')
            added=root/'RootInteractive/MLpipeline/tests/test_new.py';added.write_text('def test_new(): assert True\n')
            tracked=root/dirs[1]/'test_smoke.py';tracked.write_text('def test_ok(): assert 1 == 1\n')
            env=dict(os.environ);env['RootInteractive_DOCS']=str(root/'docs')
            run=subprocess.run(['bash','run_tests.sh','--quick','-n','1'],cwd=d,env=env,capture_output=True,text=True)
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
            failed=subprocess.run(['bash','run_tests.sh','--quick','-n','1'],cwd=d,env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0,failed.stdout+failed.stderr)
            with zipfile.ZipFile(archive) as z:
                self.assertIn(b'SyntaxError',z.read('Tools_test.log'))
                self.assertEqual(z.read('source/RootInteractive/Tools/test_smoke.py'),tracked.read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
