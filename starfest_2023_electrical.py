"""
Draws 2 maps of the 2023 Starfest electrical grid
"""

import cv2      # type: ignore
import copy
import numpy as np
import numpy.typing as npt
from typing import Optional, Callable, Any, Iterable, Tuple, TypeAlias, List
import typing

# Closest I can get for mypy types on the numpy stuff :(
# email andrew.brownbill@gmail.com if you know how to do this?
Vector:         TypeAlias = 'np.ndarray[Any, Any ]'
Matrix:         TypeAlias = 'np.ndarray[Any, Any ]'
Image:          TypeAlias = 'np.ndarray[Any, Any ]'

# For type checking with mypy
Pixel=          tuple[int, int]
Color=          tuple[int, int, int]
CoordToPixel=   Callable[ [Vector], Pixel]

class MapFeature:
  def __init__( self, 
      name:           str,
      feature_type:   str, 
      long:           float, 
      lat:            float, 
      map_name:       str, 
      destination:    Optional[str], 
      usage:          Optional[int], 
      pixel:          Optional[Pixel] 
    ):
    """
    Features on the electrical grid maps

    name:         The name of the feature.  i.e., "L1", "Post 1"
    type:         The type of feature.  i.e., "Spool", "Mat"
    coord:        The GPS co-ordinate of the feature
    map_name:     The map the feature is on "i.e., "map1", "map2"
    destination:  What does the feature connect to?  Used for line drawing
                  for power connections, the tent, and the road
    users_2023:   Number of users a spool had at Starfest 2023
    pixel:        Location of feature on the map image, if known.
    """
    self.name         = name
    self.type         = feature_type
    self.coord        = np.array( [long, lat], np.float64 )
    self.map_name     = map_name
    self.destination  = destination
    self.users_2023   = usage
    if pixel:
      (x, y) = pixel
      self.pixel    = np.array( [x, y], np.float64 )

# Type checking that depends on MapFeature
MapFeatures =     List[ MapFeature ]    # Explicitely list and not iterable
FeatureFilter =   Callable[ [MapFeature], bool ]
FeatureDrawer=    Callable[ [MapFeature, Pixel, Optional[Pixel]], None]

MAP_FEATURES = [
  #
  # These "features" bind map co-ordinates to their pixel locations on the mao image
  # They are not intended to be drawn
  #
  MapFeature("map1_mark0", "Marker",  44.074086,  -80.837580,    "map1", None, None, (1144, 560)),
  MapFeature("map1_mark1", "Marker",  44.075378,  -80.841208,    "map1", None, None, (62, 26)),
  MapFeature("map1_mark2", "Marker",  44.073256,  -80.840810,    "map1", None, None, (180, 905)),
  MapFeature("map2_mark0", "Marker",  44.075212,  -80.838710,    "map2", None, None, (20, 665)),
  MapFeature("map2_mark1", "Marker",  44.075634,  -80.836970,    "map2", None, None, (714, 432 )),
  MapFeature("map2_mark2", "Marker",  44.075914,  -80.838481,    "map2", None, None, (109, 272)),

  #
  # Features that are drawn are below this line
  #

  # The Tent corners
  MapFeature("T0",   "Tent",    44.074405, -80.840559,   "map1", "T3", None, None),
  MapFeature("T1",   "Tent",    44.074513, -80.840593,   "map1", "T0", None, None),
  MapFeature("T2",   "Tent",    44.074568, -80.840197,   "map1", "T1", None, None),
  MapFeature("T3",   "Tent",    44.074453, -80.840174,   "map1", "T2", None, None),

  # Power Sources
  MapFeature("Post 1",  "Source",  44.074123, -80.839866,   "map1", None, None, None),
  MapFeature("Post 2",  "Source",  44.074317, -80.839548,   "map1", None, None, None),
  MapFeature("Post 3",  "Source",  44.074534, -80.839359,   "map1", None, None, None),
  MapFeature("Rec",     "Source",  44.074000, -80.840586,   "map1", None, None, None),
  MapFeature("RecTurn", "Source",  44.073910, -80.840562,   "map1", "Rec", None, None),
  MapFeature("DevLand", "Source",  44.075484, -80.838499,   "map2", None, None, None),
  MapFeature("Washroom","Source",  44.074229, -80.841269,   "map1", None, None, None),
  
  # All spools, in mostly numeric order.
  # 
  # UnB  = Unlabeled Spool with box 
  # UnNB = Unlabeled Spool without box
  #
  MapFeature("L1-L4","Spool",   44.074388, -80.840559,   "map1", "Washroom",None,   None),
  MapFeature("L5",   "Spool",   44.073819, -80.840144,   "map1", "RecTurn", 3,    None),
  MapFeature("L6",   "Spool",   44.074449, -80.840571,   "map1", "Washroom",0,    None),
  MapFeature("L7",   "Spool",   44.076034, -80.838061,   "map2", "Mat0",    2,    None),
  MapFeature("L8",   "Spool",   44.075628, -80.837688,   "map2", "Mat3",    5,    None),
  MapFeature("L9",   "Spool",   44.073794, -80.839826,   "map1", "RecTurn", 5,    None),
  MapFeature("L10",  "Spool",   44.075617, -80.837186,   "map2", "L13",     6,    None),
  MapFeature("L11",  "Spool",   44.074678, -80.839005,   "map1", "Post 3",  6,    None),
  MapFeature("L13",  "Spool",   44.075422, -80.837622,   "map2", "Mat3",    4,    None),
  MapFeature("L14",  "Spool",   44.074679, -80.839006,   "map1", "Post 3",  0,    None),
  MapFeature("L16",  "Spool",   44.073448, -80.839804,   "map1", "L5",      6,    None),
  MapFeature("L17",  "Spool",   44.074730, -80.838445,   "map1", "L14",     6,    None ),
  MapFeature("L18",  "Spool",   44.074076, -80.839474,   "map1", "Post 1",  6,    None),
  MapFeature("L19",  "Spool",   44.075393, -80.837898,   "map2", "Mat3",    3,    None),
  MapFeature("L20",  "Spool",   44.074476, -80.839055,   "map1", "Post 3",  5,    None),
  MapFeature("L22",  "Spool",   44.074370, -80.839333,   "map1", "Post 2",  4,    None),
  MapFeature("UnB",  "Spool",   44.075177, -80.837342,   "map2", "L19",     1,    None),
  MapFeature("UnNB", "Spool",   44.075834, -80.836807,   "map2", "L8",      7,    None),

  # heavy black rubber mats that protect lines when they cross roads
  MapFeature("Mat0", "Mat",     44.075896, -80.838303,   "map2", "Mat1",    None, None),
  MapFeature("Mat1", "Mat",     44.075774, -80.838389,   "map2", "Mat2",    None, None),
  MapFeature("Mat2", "Mat",     44.075555, -80.838434,   "map2", "DevLand", None, None),
  MapFeature("Mat3", "Mat",     44.075458, -80.838376,   "map2", "DevLand", None, None),
  MapFeature("Mat4", "Mat",     44.074578, -80.840731,   "map1", None,      None, None),
  MapFeature("Mat5", "Mat",     44.074298, -80.841127,   "map1", None,      None, None),
  MapFeature("Mat6", "Mat",     44.073878, -80.840530,   "map1", None,      None, None),

  # Approximate location of the chalk road that goes through the south field
  # RC = Road center.

  MapFeature("RC0",   "Road",    44.075122, -80.838502,  "map1", None,      None, None),
  MapFeature("RC1",   "Road",    44.074498, -80.838752,  "map1", "RC0",     None, None),
  MapFeature("RC2",   "Road",    44.073886, -80.839380,  "map1", "RC1",     None, None),
  MapFeature("RC3",   "Road",    44.073969, -80.839934,  "map1", "RC2",     None, None),
  MapFeature("RC4",   "Road",    44.073790, -80.840477,  "map1", "RC3",     None, None),
]

def compute_coord_to_pixel_function(
    entry1:   MapFeature, 
    entry2:   MapFeature, 
    basis:    MapFeature 
  ) -> CoordToPixel:
  """
  Computes a function that maps coordinates (degrees) to pixel
  locations on a map given three MapFeature records.
  
  1. Compute degree and pixel vectors from basis to entry
  2. Find a 2x2 matrix, M, to map degree vectors to pixel vectors
  3. Helper function to map degrees to pixels given M and the basis
  4. Capture M and the Basis in a lambda and return it
  """

  """
  1. Compute degree and pixel vectors from basis to entry
  """
  deg_vec_0: Vector = entry1.coord - basis.coord
  deg_vec_1: Vector = entry2.coord - basis.coord
  pixel_vec_0 = entry1.pixel - basis.pixel
  pixel_vec_1 = entry2.pixel - basis.pixel

  """
  2. Find a 2x2 matrix, M, to map degree vectors to pixel vectors

  M * deg_vec_0 = pixel_vec_0
  M * deg_vec_1 = pixel_vec_1

  Convert deg_vec_0/deg_vec_1 and pixel_vec_0/pixel_vec_1 to 2x2 matrixes

  M * M_Degree = M_Pixel
  M * M_Degree * M_Degree_inv = M_Pixel * M_Degree_Inv
  M = MPixel * M_Degree_Inv
  """
  M_Degree     = np.transpose(np.array( [deg_vec_0, deg_vec_1] ))
  M_Pixel      = np.transpose(np.array( [pixel_vec_0, pixel_vec_1] ))
  M_Degree_Inv = np.linalg.inv( M_Degree ) #type: ignore[no-untyped-call]
  M = np.matmul( M_Pixel, M_Degree_Inv )

  """
  3. Helper function to map degrees to pixels given M and the basis
  """
  def coord_to_pixel_helper( 
      coord : Vector,
      M     : Matrix,
      basis : MapFeature 
    ) -> Pixel:
    pixel = M.dot(coord - basis.coord) + basis.pixel
    return ( int( pixel[0] ), int( pixel[1] ) )

  """
  4. Capture M and the Basis in a lambda and return it
  """
  return lambda coord : coord_to_pixel_helper( coord, M, basis )


def find_map_entry( 
    features:   MapFeatures, 
    name:       str 
  ) -> MapFeature:
  """
  Finds a named feature in a list of features.
  
  Note: The linear search is inefficint but the data set is small
  """
  try:
    return next(filter(lambda feature: feature.name == name, features))
  except StopIteration:
    assert False, "Map Feature " + name + " not found"

def get_destination_pixel( 
    map_features:       MapFeatures, 
    destination_name:   Optional[str], 
    coord_to_pixel:     CoordToPixel 
  ) -> Optional[Pixel]:
  """
  Given a destination name, compute the pixel location of the destination
  """
  if destination_name:
    destination_entry = find_map_entry( map_features, destination_name )
    return coord_to_pixel( destination_entry.coord )
  return None


def draw_features( 
    map_features:     MapFeatures, 
    coord_to_pixel:   CoordToPixel,
    feature_filter:   FeatureFilter, 
    feature_drawer:   FeatureDrawer 
  ) -> None:
  """
  Draw some map features
  
  map_features    : complete list of features for the map we're drawing
  coord_to_pixel  : function to map co-ordinates to pixel locations
  feature_filter  : function that returns true if we're to draw this feature
  feature_drawer  : The function that actually draws the feature
  """
  for feature in filter( feature_filter, map_features):
      pixel = coord_to_pixel( feature.coord )
      destination_pixel = get_destination_pixel( map_features, feature.destination, coord_to_pixel )
      feature_drawer( feature, pixel, destination_pixel )

def filter_for_type(filter_type : str) -> FeatureFilter:
  """
  Create a function that returns true if a feature is a particular type
  """
  return lambda feature: feature.type == filter_type


def filter_for_line(filter_type : str) -> FeatureFilter:
  """
  Create a function that returns true if a feature is a type and has a destination
  """
  return lambda feature: feature.type == filter_type and feature.destination is not None


def create_line_drawer( 
    image:        Image, 
    color:        Color, 
    line_width:   int 
  ) -> FeatureDrawer:
  """
  Create a function that draws a line for features with a destination
  """
  return lambda feature, pixel, dest_pixel : cv2.line( 
      image, pixel, dest_pixel, color, line_width )


def create_label_drawer( 
    image:    Image, 
    color:    Color 
  ) -> FeatureDrawer:
  """
  Create a function that uses the feature name to draw a label
  """
  return lambda feature, pixel, dest_pixel: cv2.putText( 
        image, feature.name, 
        (pixel[0]+2, pixel[1]+9), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA, False ) 


def draw_crosshair( 
    image:    Image, 
    point:    Pixel, 
    color:    Color
  ) -> None:
  """
  Draws a small crosshair at point
  """
  line_width = 4
  cross_size = 3
  ( x, y ) = point
  lower_left  = ( x - cross_size, y - cross_size )
  lower_right = ( x + cross_size, y - cross_size )
  upper_left  = ( x - cross_size, y + cross_size )
  upper_right = ( x + cross_size, y + cross_size )

  cv2.line( image, lower_left, upper_right, color, line_width ) 
  cv2.line( image, lower_right, upper_left, color, line_width ) 


def create_crosshair_drawer( 
    image:            Image, 
    color:            Color 
  ) -> FeatureDrawer:
  """
  Create a function that draws a crosshair
  """
  return lambda feature, pixel, dest_pixel: draw_crosshair( image, pixel, color ) 


def draw_spools( 
    image:            Image, 
    features:         MapFeatures, 
    coord_to_pixel:   CoordToPixel 
  ) -> None:
  """
  Draw all the spools.
  """
  off_white = ( 228, 228, 228 )
  blue = ( 255, 0, 0 )
  draw_features( features, coord_to_pixel, filter_for_type("Spool"), create_crosshair_drawer( image, blue ))
  draw_features( features, coord_to_pixel, filter_for_type("Spool"), create_label_drawer( image, off_white ))


def draw_mats( 
    image:            Image, 
    features:         MapFeatures, 
    coord_to_pixel:   CoordToPixel 
  ) -> None:
  """
  Draw all the road mats.
  """
  off_white = ( 228, 228, 228 )
  blue      = ( 255, 0, 0 )
  draw_features( features, coord_to_pixel, filter_for_type("Mat"), create_crosshair_drawer( image, blue ))
  draw_features( features, coord_to_pixel, filter_for_type("Mat"), create_label_drawer( image, off_white ))

def draw_lines( 
    image:            Image, 
    features:         MapFeatures, 
    coord_to_pixel:   CoordToPixel, 
    feature_types:    Iterable[str], 
    color:            Color, 
    alpha:            float, 
    width:            int = 4
  ) -> None:
  """
  Draw lines for a set of feature types.  Allows blending.
  
  image:            The image we're drawing onto
  features:         Feature list
  coord_to_pixel:   Function that convers GPS coordinate to pixels
  color:            Line color
  alpha:            Transparency.  0-1,  1 = full opaque
  width:            Line width
  """

  beta = 1-alpha
  overlay = image.copy()  # draw lines to an overlay at 100% transparency, then blend the overlay and the original image
  for feature_type in feature_types:
    draw_features( features, coord_to_pixel, filter_for_line(feature_type), create_line_drawer( overlay, color, width ))
  image = cv2.addWeighted( overlay, alpha, image, beta, 0, image )

def draw_roads( 
    image:          Image, 
    features:       MapFeatures, 
    coord_to_pixel: CoordToPixel 
  ) -> None:
  """
  Draw all the roads
  """
  cyan = (128,128,0)
  alpha = .30              # draw roads with 30% transparency
  road_width = 20
  draw_lines( image, features, coord_to_pixel, ["Road"], cyan, alpha, road_width )

def draw_tent( 
    image:          Image, 
    features:       MapFeatures, 
    coord_to_pixel: CoordToPixel 
  ) -> None:
  """
  Draw the tent
  """
  white = (255,255,255)
  alpha = .50             # draw tent at 50 percent tranparency
  draw_lines( image, features, coord_to_pixel, ["Tent"], white, alpha )


def draw_electric_cords( 
    image:          Image, 
    features:       MapFeatures, 
    coord_to_pixel: CoordToPixel 
  ) -> None:
  """
  Draw the electric grid
  
  Grid lines can go from spool to spool, spool to mat, mat to source, and source to source.
  source to source was used at the rec hall, where we did a 90 degree turn to go around
  the food truck in 2023.
  """
  blue = (255, 0, 0 )
  alpha = .50             # draw tent at 50 percent tranparency
  draw_lines( image, features, coord_to_pixel, ["Spool", "Source", "Mat" ], blue, alpha )


def draw_all_features( 
    image:          Image, 
    map_features:   MapFeatures, 
    coord_to_pixel: CoordToPixel 
  ) -> None:
  """
  Draw everything on a map.
  
  order matters. We want the labels for the mats and spools drawn
  after the road, electric cords, and tent so they show up well.
  """
  draw_roads         ( image, map_features, coord_to_pixel )
  draw_electric_cords( image, map_features, coord_to_pixel )
  draw_tent          ( image, map_features, coord_to_pixel )
  draw_mats          ( image, map_features, coord_to_pixel )
  draw_spools        ( image, map_features, coord_to_pixel )


def draw_map( map_name: str ) -> None:
  """
  Draws one of the electric grid maps
  
  1.  Load the map template from disk
  2.  Find all the features for this map
  3.  Compute a function that maps GPS coordinates to pixels
  4.  Draw all the features for our map
  5.  Save the new map
  """

  """
  1.  Load the map template from disk
  """
  image = cv2.imread(map_name + ".png")
  
  """
  2.  Find all the features for this map
  """
  map_features = list(filter( lambda x: x.map_name == map_name, MAP_FEATURES ))

  """
  3.  Compute a function that maps GPS coordinates to pixels
  """
  map_coord_to_pixel = compute_coord_to_pixel_function(
      find_map_entry( map_features, map_name + "_mark0" ),
      find_map_entry( map_features, map_name + "_mark1" ),
      find_map_entry( map_features, map_name + "_mark2" ))

  """
  4.  Draw all the features for our map
  """
  draw_all_features( image, map_features, map_coord_to_pixel ) 

  """
  5.  Save the new map
  """
  cv2.imwrite(map_name + "_spool.png", image )


def main() -> None:
  """
  Draws both the Starfest electric grid maps
  """
  draw_map( "map1" )    # South Field
  draw_map( "map2" )    # North Field

if __name__ == "__main__":
  main()


