import os
import json
import argparse
import pdb

"""
This simple script aims to fix the entwine output's src code bug in a hard-code manner.
entwine 2.2 does not output "authority" and "horizontal" attributes in EPT json file's "src" section.
"""


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('ept_json_path', help="EPT data json path")
    args = parser.parse_args()
    args.ept_json_path = os.path.abspath(args.ept_json_path)
    return args


def main():
    args = config_parser()
    print(args)
    assert os.path.exists(args.ept_json_path), "EPT json at {:s} is not found!".format(args.ept_json_path)
    with open(args.ept_json_path, 'r') as f:
        data = json.load(f)

    srs_data = data['srs']
    rewrite = False if "authority" in srs_data.keys() and "horizontal" in srs_data.keys() else True
    if rewrite:
        data['srs']['authority'] = "EPSG"
        data['srs']['horizontal'] = "4978"
        with open(args.ept_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print("EPT json fix is done w/ rewriting.")
    else:
        print("EPT json fix is done w/o rewriting.")


if __name__ == "__main__":
    main()
