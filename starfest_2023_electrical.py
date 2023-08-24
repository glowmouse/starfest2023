"""
Draws 2 maps of the 2023 Starfest electrical grid
"""

import cv2
import copy
import numpy as np

class MapFeature:
  def __init__( self, name: str, feature_type: str, long: float, lat: float, map_name: str, destination:str, usage:int, pixel ):
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
    self.map_name      = map_name
    self.destination  = destination
    self.users_2023   = usage
    if pixel:
      (x, y) = pixel
      self.pixel    = np.array( [x, y], np.float64 )


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
  # RR = Road Right, RL = Road Left.
  MapFeature("RR0",   "Road",    44.075130, -80.838528,  "map1", None,      None, None),
  MapFeature("RR1",   "Road",    44.074479, -80.838801,  "map1", "RR0",     None, None),
  MapFeature("RR2",   "Road",    44.073924, -80.839380,  "map1", "RR1",     None, None),
  MapFeature("RR3",   "Road",    44.073988, -80.839955,  "map1", "RR2",     None, None),
  MapFeature("RR4",   "Road",    44.073822, -80.840468,  "map1", "RR3",     None, None),

  MapFeature("RL0",   "Road",    44.075114, -80.838477,  "map1", None,      None, None),
  MapFeature("RL1",   "Road",    44.074518, -80.838703,  "map1", "RL0",     None, None),
  MapFeature("RL2",   "Road",    44.073849, -80.839371,  "map1", "RL1",     None, None),
  MapFeature("RL3",   "Road",    44.073951, -80.839914,  "map1", "RL2",     None, None),
  MapFeature("RL4",   "Road",    44.073759, -80.840486,  "map1", "RL3",     None, None),
]

def compute_coord_to_pixel_function(entry1 : MapFeature, entry2 : MapFeature, basis: MapFeature ):
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
  deg_vec_0 = entry1.coord - basis.coord
  deg_vec_1 = entry2.coord - basis.coord
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
  M_Degree_Inv = np.linalg.inv( M_Degree )
  M = np.matmul( M_Pixel, M_Degree_Inv )

  """
  3. Helper function to map degrees to pixels given M and the basis
  """
  def coord_to_pixel_helper( coord, M, basis ):
    pixel = M.dot(coord - basis.coord) + basis.pixel
    return ( int( pixel[0] ), int( pixel[1] ) )

  """
  4. Capture M and the Basis in a lambda and return it
  """
  return lambda coord : coord_to_pixel_helper( coord, M, basis )


def find_map_entry( features, name: str ):
  """
  Finds a named feature in a list of features.
  
  Note: The linear search is inefficint but the data set is small
  """
  for feature in features:
    if name == feature.name:
      return feature
  print("Cannot find " + name )
  assert( False )


def get_destination_pixel( map_features, destination_name : str, coord_to_pixel ):
  """
  Given a destination name, compute the pixel location of the destination
  """
  if destination_name:
    destination_entry = find_map_entry( map_features, destination_name )
    return coord_to_pixel( destination_entry.coord )
  return None


def draw_features( map_features, coord_to_pixel, feature_filter, feature_drawer ):
  """
  Draw some map features
  
  map_features    : complete list of features for the map we're drawing
  coord_to_pixel  : function to map co-ordinates to pixel locations
  feature_filter  : function that returns true if we're to draw this feature
  feature_drawer  : The function that actually draws the feature
  """
  for feature in map_features:
    pixel = coord_to_pixel( feature.coord )
    destination_pixel = get_destination_pixel( map_features, feature.destination, coord_to_pixel )
    if feature_filter(feature, pixel, destination_pixel ):
      feature_drawer( feature, pixel, destination_pixel )


def filter_for_type( filter_type ):
  """
  Create a function that returns true if a feature is a particular type
  """
  return lambda feature, pixel, dest_pixel: feature.type == filter_type


def filter_for_line( filter_type ):
  """
  Create a function that returns true if a feature is a type and has a destination
  """
  return lambda feature, pixel, dest_pixel: feature.type == filter_type and dest_pixel


def create_line_drawer( image, color ):
  """
  Create a function that draws a line for features with a destination
  """
  line_width = 4
  return lambda feature, pixel, dest_pixel : cv2.line( 
      image, pixel, dest_pixel, color, line_width )


def create_label_drawer( image, color ):
  """
  Create a function that uses the feature name to draw a label
  """
  return lambda feature, pixel, dest_pixel: cv2.putText( 
        image, feature.name, 
        (pixel[0]+2, pixel[1]+9), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA, False ) 


def draw_crosshair( image, point, color ):
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


def create_crosshair_drawer( image, color ):
  """
  Create a function that draws a crosshair
  """
  return lambda feature, pixel, dest_pixel: draw_crosshair( image, pixel, color ) 


def draw_spools( image, features, coord_to_pixel ):
  """
  Draw all the spools.
  """
  off_white = ( 228, 228, 228 )
  blue = ( 255, 0, 0 )
  draw_features( features, coord_to_pixel, filter_for_type("Spool"), create_crosshair_drawer( image, blue ))
  draw_features( features, coord_to_pixel, filter_for_type("Spool"), create_label_drawer( image, off_white ))


def draw_mats( image, features, coord_to_pixel ):
  """
  Draw all the road mats.
  """
  off_white = ( 228, 228, 228 )
  blue      = ( 255, 0, 0 )
  draw_features( features, coord_to_pixel, filter_for_type("Mat"), create_crosshair_drawer( image, blue ))
  draw_features( features, coord_to_pixel, filter_for_type("Mat"), create_label_drawer( image, off_white ))


def draw_roads( image, features, coord_to_pixel ):
  """
  Draw all the roads
  """
  cyan = (128,128,0)
  draw_features( features, coord_to_pixel, filter_for_line("Road"), create_line_drawer( image, cyan ))


def draw_tent( image, features, coord_to_pixel ):
  """
  Draw the tent
  """
  white = (255,255,255)
  draw_features( features, coord_to_pixel, filter_for_line("Tent"), create_line_drawer( image, white ))


def draw_electric_cords( image, features, coord_to_pixel ):
  """
  Draw the electric grid
  
  Grid lines can go from spool to spool, spool to mat, mat to source, and source to source.
  source to source was used at the rec hall, where we did a 90 degree turn to go around
  the food truck in 2023.
  """
  blue = (255, 0, 0 )
  draw_features( features, coord_to_pixel, filter_for_line("Spool"), create_line_drawer( image, blue ))
  draw_features( features, coord_to_pixel, filter_for_line("Source"), create_line_drawer( image, blue ))
  draw_features( features, coord_to_pixel, filter_for_line("Mat"), create_line_drawer( image, blue ))


def draw_all_features( image, map_features, coord_to_pixel ):
  """
  Draw everything on a map.
  
  order matters. We want the labels for the mats and spools drawn
  after the road, electric cords, and tent so they show up well.
  """
  draw_electric_cords( image, map_features, coord_to_pixel )
  draw_roads         ( image, map_features, coord_to_pixel )
  draw_tent          ( image, map_features, coord_to_pixel )
  draw_mats          ( image, map_features, coord_to_pixel )
  draw_spools        ( image, map_features, coord_to_pixel )


def filter_features(map_features, map_name):
  """
  Given an existing feature list, create a new feature list
  filtered by mapname and return it.
  """
  filtered_map_features = []
  for feature in map_features:
    if feature.map_name == map_name:
      filtered_map_features.append( feature )
  return filtered_map_features 


def draw_map( map_name ):
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
  map_features = filter_features( MAP_FEATURES, map_name )

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


def main():
  """
  Draws both the Starfest electric grid maps
  """
  draw_map( "map1" )    # South Field
  draw_map( "map2" )    # North Field

if __name__ == "__main__":
  main()


