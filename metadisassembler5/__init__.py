#!/usr/bin/env python
from .MetaDisassembler5 import MetaDisassembler5

def main():
    md = MetaDisassembler5()
    md._get_args()
    md.input_query(md.query)
    md.disassemble()

    return True

if __name__ == "__main__":
    main()
