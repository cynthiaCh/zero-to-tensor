import unittest

from experiments.fourier_signal_demo import (
    find_top_frequencies,
    make_mixed_signal,
    spectrum_strength,
)


class FourierSignalDemoTest(unittest.TestCase):
    def test_fft_finds_the_two_frequencies_in_mixed_signal(self):
        sample_count = 128
        signal = make_mixed_signal(sample_count=sample_count)
        frequencies, strengths = spectrum_strength(signal)

        top = find_top_frequencies(frequencies, strengths, count=2)

        self.assertEqual([item["frequency"] for item in top], [3, 15])
        self.assertGreater(top[0]["strength"], top[1]["strength"])


if __name__ == "__main__":
    unittest.main()
