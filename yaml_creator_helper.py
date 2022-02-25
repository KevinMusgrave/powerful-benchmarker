import argparse
import string


def main(cfg):
    base_string = """
- - python launch_one.py --src_domains art --target_domains clipart
  - *A
  - *common
- - python launch_one.py --src_domains art --target_domains product
  - *A
  - *common
- - python launch_one.py --src_domains art --target_domains real
  - *A
  - *common
- - python launch_one.py --src_domains clipart --target_domains art
  - *A
  - *common
- - python launch_one.py --src_domains clipart --target_domains product
  - *A
  - *common
- - python launch_one.py --src_domains clipart --target_domains real
  - *A
  - *common
- - python launch_one.py --src_domains product --target_domains art
  - *A
  - *common
- - python launch_one.py --src_domains product --target_domains clipart
  - *A
  - *common
- - python launch_one.py --src_domains product --target_domains real
  - *A
  - *common
- - python launch_one.py --src_domains real --target_domains art
  - *A
  - *common
- - python launch_one.py --src_domains real --target_domains product
  - *A
  - *common
- - python launch_one.py --src_domains real --target_domains clipart
  - *A
  - *common
"""

    out_string = f"{base_string}\n"
    for i in range(1, 5):
        curr_char = string.ascii_uppercase[i]
        new_block = base_string.replace("*A", f"*{curr_char}")
        out_string += f"{new_block}\n"
    if cfg.save_to_file:
        with open(cfg.save_to_file, "w") as f:
            f.write(out_string)
    else:
        print(out_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--save_to_file",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
