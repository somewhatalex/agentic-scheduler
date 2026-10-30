"""CloudLab profile: one bare-metal node with the project's persistent dataset at /mydata.

IMPORTANT: instantiate this on the Utah cluster. A remote blockstore can only
be attached to nodes on the same cluster as the dataset, which lives on Utah.

Everything on the node's local disk is wiped when the experiment expires; only
/mydata (the long-term dataset) survives between experiments.
"""

import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()

# Fill in the URN from the dataset's info page in the CloudLab portal
# (Storage > My Datasets > <dataset> > URN), or override it when instantiating.
pc.defineParameter(
    "dataset_urn",
    "Persistent dataset URN",
    portal.ParameterType.STRING,
    "urn:publicid:IDN+utah.cloudlab.us:michigan-bigdata-pg0+stdataset+agentic-sched-data",
)
# Only the dataset owner can write. Leave this on unless you are downloading or
# processing data; a dataset can only be mounted read-write by one experiment at a time.
pc.defineParameter(
    "readonly",
    "Mount /mydata read-only",
    portal.ParameterType.BOOLEAN,
    True,
)
pc.defineParameter(
    "hw_type",
    "Hardware type (leave blank for any)",
    portal.ParameterType.NODETYPE,
    "",
)
params = pc.bindParameters()

request = pc.makeRequestRSpec()

node = request.RawPC("node")
node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD"
if params.hw_type:
    node.hardware_type = params.hw_type
iface = node.addInterface()

rbs = request.RemoteBlockstore("dataset", "/mydata")
rbs.dataset = params.dataset_urn
rbs.readonly = params.readonly

link = request.Link("dataset-link")
link.addInterface(iface)
link.addInterface(rbs.interface)
link.best_effort = True
link.vlan_tagging = True

pc.printRequestRSpec(request)
