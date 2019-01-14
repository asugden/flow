# Filter out annoying messages about binary incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Sub-folders
from . import classifier, metadata, misc
# Individual files
from . import classify2p, config, glm, outfns, _parseargv, paths, trace2p, xday
# Pull in important classes for easy use
from .metadata.sorters import Mouse, Date, Run, MouseSorter, DateSorter, RunSorter, DatePairSorter
