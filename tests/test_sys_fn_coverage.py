import math
import os
import sys
import tempfile
import unittest

from klongpy import KlongInterpreter, KlongException
from tests.backend_compat import requires_strings
from klongpy.core import KGChannel, KGChannelDir, KGSym, KGLambda
from klongpy.sys_fn import (
    eval_sys_delete_file,
    eval_sys_from_channel,
    eval_sys_flush,
    eval_sys_input_channel,
    eval_sys_load,
    eval_sys_more_input,
    eval_sys_module,
    eval_sys_output_channel,
    eval_sys_to_channel,
    eval_sys_process_clock,
    eval_sys_python,
    eval_sys_python_call,
    eval_sys_python_from,
    eval_sys_python_attribute,
    eval_sys_read,
    eval_sys_read_lines,
    eval_sys_system,
    import_directory_module,
    import_file_module,
    import_module_from_sys,
    _handle_import,
)


class TestDeleteFileErrorHandling(unittest.TestCase):
    def test_delete_nonexistent_file(self):
        with self.assertRaises(RuntimeError) as ctx:
            eval_sys_delete_file("/nonexistent/path/file.txt")
        self.assertIn("does not exist", str(ctx.exception))


class TestFromChannelErrorHandling(unittest.TestCase):
    def test_from_channel_with_output_channel(self):
        klong = KlongInterpreter()
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            fname = f.name
        try:
            with eval_sys_output_channel(fname) as out_chan:
                with self.assertRaises(RuntimeError) as ctx:
                    eval_sys_from_channel(klong, out_chan)
                self.assertIn("output channel cannot be used input", str(ctx.exception))
        finally:
            os.unlink(fname)


class TestFlushErrorHandling(unittest.TestCase):
    def test_flush_input_channel(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            fname = f.name
            f.write("test")
        try:
            with eval_sys_input_channel(fname) as in_chan:
                with self.assertRaises(RuntimeError) as ctx:
                    eval_sys_flush(in_chan)
                self.assertIn("input channel cannot be flushed", str(ctx.exception))
        finally:
            os.unlink(fname)


class TestMoreInputErrorHandling(unittest.TestCase):
    def test_more_input_with_output_channel(self):
        klong = KlongInterpreter()
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            fname = f.name
        try:
            with eval_sys_output_channel(fname) as out_chan:
                klong['.sys.cin'] = out_chan
                with self.assertRaises(RuntimeError) as ctx:
                    eval_sys_more_input(klong)
                self.assertIn("output channel cannot be used for input", str(ctx.exception))
        finally:
            os.unlink(fname)


class TestToChannelErrorHandling(unittest.TestCase):
    def test_to_channel_with_input_channel(self):
        klong = KlongInterpreter()
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            fname = f.name
            f.write("test")
        try:
            with eval_sys_input_channel(fname) as in_chan:
                with self.assertRaises(RuntimeError) as ctx:
                    eval_sys_to_channel(klong, in_chan)
                self.assertIn("input channel cannot be a to channel", str(ctx.exception))
        finally:
            os.unlink(fname)


class TestProcessClock(unittest.TestCase):
    def test_process_clock(self):
        klong = KlongInterpreter()
        result = eval_sys_process_clock(klong)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)


class TestModule(unittest.TestCase):
    def test_module_start_and_stop(self):
        klong = KlongInterpreter()
        eval_sys_module(klong, "testmod")
        eval_sys_module(klong, 0)  # Stop module
        eval_sys_module(klong, "testmod2")
        eval_sys_module(klong, "")  # Stop module with empty string


class TestLoadFileNotFound(unittest.TestCase):
    def test_load_nonexistent_file(self):
        klong = KlongInterpreter()
        with self.assertRaises(FileNotFoundError):
            eval_sys_load(klong, "/nonexistent/path/file.kg")

    def test_load_with_klongpath(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "test.kg")
            with open(fname, "w") as f:
                f.write("42")
            old_path = os.environ.get('KLONGPATH')
            try:
                os.environ['KLONGPATH'] = td
                result = eval_sys_load(klong, "test")
                self.assertEqual(result, 42)
            finally:
                if old_path:
                    os.environ['KLONGPATH'] = old_path
                elif 'KLONGPATH' in os.environ:
                    del os.environ['KLONGPATH']


class TestPythonImport(unittest.TestCase):
    def test_python_import_non_string_raises(self):
        klong = KlongInterpreter()
        with self.assertRaises(RuntimeError) as ctx:
            eval_sys_python(klong, 123)
        self.assertIn("module name must be a string", str(ctx.exception))

    def test_python_from_non_string_module_raises(self):
        klong = KlongInterpreter()
        with self.assertRaises(RuntimeError) as ctx:
            eval_sys_python_from(klong, 123, "sqrt")
        self.assertIn("module name must be a string", str(ctx.exception))

    def test_python_from_non_string_items_raises(self):
        klong = KlongInterpreter()
        with self.assertRaises(RuntimeError) as ctx:
            eval_sys_python_from(klong, "math", [123])
        self.assertIn("from list entry must be a string", str(ctx.exception))


class TestPythonCall(unittest.TestCase):
    def test_python_call_list_form(self):
        klong = KlongInterpreter()
        # Pass the actual object, not a string key
        result = eval_sys_python_call(klong, [math, "sqrt"], [64], {})
        self.assertEqual(result, 8.0)

    def test_python_call_list_form_with_symbol(self):
        klong = KlongInterpreter()
        klong["math_mod"] = math
        # Use KGSym for lookup
        result = eval_sys_python_call(klong, [KGSym("math_mod"), "sqrt"], [64], {})
        self.assertEqual(result, 8.0)

    def test_python_call_invalid_function_raises(self):
        klong = KlongInterpreter()
        with self.assertRaises(KlongException) as ctx:
            eval_sys_python_call(klong, [math, "nonexistent"], [1], {})
        self.assertIn("not found", str(ctx.exception))

    def test_python_call_non_dict_kwargs_raises(self):
        klong = KlongInterpreter()
        klong["fn"] = lambda x: x
        with self.assertRaises(KlongException) as ctx:
            eval_sys_python_call(klong, "fn", [1], "not_a_dict")
        self.assertIn("must be a dictionary", str(ctx.exception))

    def test_python_call_scalar_arg(self):
        klong = KlongInterpreter()
        klong["fn"] = lambda x: x + 1
        result = eval_sys_python_call(klong, "fn", 2, {})
        self.assertEqual(result, 3)

    def test_python_call_non_callable(self):
        klong = KlongInterpreter()
        klong["val"] = 42
        result = eval_sys_python_call(klong, "val", [], {})
        self.assertEqual(result, 42)

    def test_python_call_list_non_callable(self):
        klong = KlongInterpreter()
        # Pass the actual object
        result = eval_sys_python_call(klong, [math, "pi"], [], {})
        self.assertAlmostEqual(result, 3.14159, places=4)


class TestPythonAttribute(unittest.TestCase):
    def test_python_attribute_non_string_name_raises(self):
        klong = KlongInterpreter()
        with self.assertRaises(KlongException) as ctx:
            eval_sys_python_attribute(klong, math, 123)
        self.assertIn("attribute name must be a string", str(ctx.exception))

    def test_python_attribute_not_found_raises(self):
        klong = KlongInterpreter()
        with self.assertRaises(KlongException) as ctx:
            eval_sys_python_attribute(klong, math, "nonexistent")
        self.assertIn("not found", str(ctx.exception))

    def test_python_attribute_success(self):
        klong = KlongInterpreter()
        result = eval_sys_python_attribute(klong, math, "pi")
        self.assertAlmostEqual(result, 3.14159, places=4)


class TestReadLines(unittest.TestCase):
    @requires_strings
    def test_read_lines(self):
        klong = KlongInterpreter()
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            fname = f.name
            f.write("line1\nline2\nline3\n")
        try:
            with eval_sys_input_channel(fname) as in_chan:
                eval_sys_from_channel(klong, in_chan)
                result = eval_sys_read_lines(klong)
                self.assertEqual(len(result), 3)
        finally:
            os.unlink(fname)


class TestReadEOF(unittest.TestCase):
    def test_read_eof(self):
        klong = KlongInterpreter()
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            fname = f.name
            # Empty file
        try:
            with eval_sys_input_channel(fname) as in_chan:
                eval_sys_from_channel(klong, in_chan)
                result = eval_sys_read(klong)
                self.assertIsNone(result)
                self.assertTrue(in_chan.at_eof)
        finally:
            os.unlink(fname)


class TestSystem(unittest.TestCase):
    def test_system_command(self):
        # Using a simple command that exists on both Unix and doesn't require arguments
        result = eval_sys_system("true")
        # returncode could be None for immediate return
        self.assertTrue(result is None or result == 0)


class TestImportHelpers(unittest.TestCase):
    def test_import_module_from_sys(self):
        module = import_module_from_sys("math")
        self.assertTrue(hasattr(module, "sqrt"))

    def test_import_module_from_sys_not_found(self):
        with self.assertRaises(RuntimeError):
            import_module_from_sys("nonexistent_module_xyz")

    def test_import_directory_module_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            # Directory without __init__.py
            with self.assertRaises(FileNotFoundError):
                import_directory_module(td)


class TestHandleImport(unittest.TestCase):
    def test_handle_import_non_callable(self):
        result = _handle_import(42)
        self.assertEqual(result, 42)

    def test_handle_import_lambda(self):
        fn = lambda x: x + 1
        result = _handle_import(fn)
        self.assertIsInstance(result, KGLambda)


class TestInputChannelIOError(unittest.TestCase):
    def test_input_channel_not_found(self):
        with self.assertRaises(FileNotFoundError):
            eval_sys_input_channel("/nonexistent/path.txt")


if __name__ == '__main__':
    unittest.main()
