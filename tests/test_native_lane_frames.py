"""The supervisor-side frame tool of the Pico bench firmware (native/boards/pico-bench/tools/lane_frames.py):
check values and layout of the 16-byte frame and the datagram layer, so that what a test operator sends
to a node is what the node's Rust code (native/klong-dgram) expects. No hardware and no native build needed."""
import importlib.util
import os
import random
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(os.path.dirname(HERE), "native", "boards", "pico-bench", "tools", "lane_frames.py")
spec = importlib.util.spec_from_file_location("lane_frames", PATH)
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)


class LaneFrames(unittest.TestCase):
    def test_check_values(self):
        self.assertEqual(lf.crc16(b"123456789"), 0x29B1)
        self.assertEqual(lf.crc32(b"123456789"), 0xCBF43926)

    def test_layout(self):
        f = lf.frame(lf.APP_DATA, 16, 2, 0x1234, lf.START, bytes([5, 0, 1, 1]))
        self.assertEqual(f[:10].hex(), "a55a0109" "1002" "3412" "d004")
        self.assertEqual(f[10:14], bytes([5, 0, 1, 1]))
        self.assertEqual(int.from_bytes(f[14:], "little"), lf.crc16(f[2:14]))
        self.assertEqual(lf.decode(f)["seq"], 0x1234)
        short = lf.frame(lf.APP_DATA, 16, 2, 0, lf.DATA, b"x")
        self.assertEqual(short[10:14], b"x\0\0\0")
        self.assertEqual(short[9], 1)

    def test_messages_of_every_size_round_trip_through_noise(self):
        rng = random.Random(7)
        for n in (1, 3, 4, 5, 1023, 1024):
            data = bytes(rng.randrange(256) for _ in range(n))
            frames = lf.message_frames(16, 0, 77, 1, data, ack=True)
            self.assertEqual(len(frames), -(-n // 4) + 2)
            stream = b"".join(bytes(rng.randrange(256) for _ in range(rng.randrange(20))) + f for f in frames)
            r = lf.Reassembler()
            got = [m for m in (r.feed(f) for f in lf.parse(stream)) if m]
            self.assertEqual(got, [(16, 1, data)], n)
        with self.assertRaises(ValueError):
            lf.message_frames(16, 0, 1, 1, bytes(1025))
        with self.assertRaises(ValueError):
            lf.message_frames(16, 0, 1, 1, b"")

    def test_duplicates_and_damage_are_not_delivered(self):
        data = b"hello world"
        frames = lf.message_frames(16, 0, 5, 1, data)
        r = lf.Reassembler()
        self.assertEqual([m for m in (r.feed(lf.decode(f)) for f in frames + frames) if m], [(16, 1, data)])
        damaged = list(frames)
        damaged[1] = lf.frame(lf.APP_DATA, 16, 0, 0, lf.DATA, b"jell")
        r = lf.Reassembler()
        self.assertEqual([m for m in (r.feed(lf.decode(f)) for f in damaged) if m], [])
        lost = frames[:2] + frames[3:]
        r = lf.Reassembler()
        self.assertEqual([m for m in (r.feed(lf.decode(f)) for f in lost) if m], [])


if __name__ == "__main__":
    unittest.main()
