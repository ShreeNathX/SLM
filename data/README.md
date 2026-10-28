# Training data

The notebook downloads the Tiny Shakespeare text file from the `karpathy/char-rnn` repository into `data/raw/`. Its README describes the file as a subset of Shakespeare's plays, and the file is about 1.06 MB. William Shakespeare's original works are public domain; the upstream repository's MIT license applies to its software, not to a claim over those underlying works. In India, the Copyright Act sets the ordinary term for published literary works at 60 years after the author's death; in the United States, published works before 1931 are public domain. Check the rules that apply where you use or redistribute a corpus.

- Source text: <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
- Source description: <https://github.com/karpathy/char-rnn/blob/master/Readme.md>
- India Copyright Act, section 22: <https://www.indiacode.nic.in/bitstream/123456789/1367/1/a195714.pdf>
- U.S. Copyright Office: <https://www.copyright.gov/what-is-copyright/>

The notebook records a SHA-256 hash for the exact text it downloads. It splits the source into disjoint training and validation spans before training the tokenizer; no sequence window crosses the split.

`instructions_cc0.jsonl` is a small, original educational conversation set created for this project and dedicated to [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). It is used only in the separate instruction-tuning stage. Its small size is unsuitable for teaching broad capabilities.
