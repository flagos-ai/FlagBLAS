# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass

import yaml


# Metadata template,  Each vendor needs to specialize instances of this template
@dataclass
class VendorInfoBase:
    vendor_name: str
    device_name: str
    device_query_cmd: str
    dispatch_key: str = None
    triton_extra_name: str = None


def get_tune_config(vendor_name=None, file_mode="r", file_path=None):
    BACKEND_EVENT = file_path is not None
    config = None
    try:
        if not file_path:
            vendor_name = "_" + vendor_name
            script_path = os.path.abspath(__file__)
            base_dir = os.path.dirname(script_path)
            file_path = os.path.join(base_dir, vendor_name, "tune_configs.yaml")
        else:
            file_path = os.path.join(file_path, "tune_configs.yaml")
        with open(file_path, file_mode) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        if not BACKEND_EVENT:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

    return config
