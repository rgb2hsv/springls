// toybox: runs any implemented configuration of the springls library, chosen on the command line by
// representation, domain, grid type, objects, scenario and interface tracking (or by -m method key).
//
//   toybox                                           # like enright2d: the four 2D CFL-limited methods
//   toybox --list                                    # the implemented configurations
//   toybox --domain 3d -m dweave                     # 3D DoubleWeave
//   toybox -r dweave --grid-type adaptive --interface-tracking eulerian,semi-lagrangian   # as sl-enright2d
//   toybox --scenario mesh model.ply                 # as springls-viewer
#include <common/Toybox.hpp>

int main(int argc, char *argv[]) { return toybox::Main(argc, argv, toybox::demos::Toybox()); }
