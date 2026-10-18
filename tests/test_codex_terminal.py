import unittest

from remory.codex.terminal import ResumeHint


class ResumeHintTest(unittest.TestCase):
    def test_native_footer_even_when_every_byte_is_a_separate_write(self):
        raw = b'\x1b[?25hTo continue this session, run \x1b[36mcodex resume 11111111-1111-1111-1111-111111111111\x1b[0m\r\n'
        expected = b'\x1b[?25hTo continue this session, run r-codex resume 20261013-074204-b9b17d\x1b[0m\r\n'
        for size in [1, 2, 17, len(raw)]:
            hint = ResumeHint('20261013-074204-b9b17d')
            out = b''.join(hint.feed(raw[i:i+size]) for i in range(0, len(raw), size)) + hint.finish()
            self.assertEqual(out, expected)
            self.assertTrue(hint.replaced)

    def test_terminal_frames_images_and_unrelated_text_are_unchanged(self):
        raw = b'\x1b[2J\x1b[5;4Hworking\r\n\x1bPqimage-payload\x1b\\To continue this session, run example\r\nTo continu'
        hint = ResumeHint('example')
        self.assertEqual(b''.join(hint.feed(bytes([c])) for c in raw) + hint.finish(), raw)
        self.assertFalse(hint.replaced)

    def test_named_native_session_uses_direct_wrapper_command(self):
        hint = ResumeHint('20261013-074204-b9b17d')
        raw = b'To continue this session, run codex resume, then select My project (11111111-1111-1111-1111-111111111111)\n'
        self.assertEqual(hint.feed(raw) + hint.finish(),
                         b'To continue this session, run r-codex resume 20261013-074204-b9b17d\x1b[0m\n')


if __name__ == '__main__':
    unittest.main()
