from utils import *
from plot_utils import *
import random
import os
import re
import numpy as np
import traceback

import shapely
from shapely.ops import substring


CANVAS_SIZE = 16
# Define MARGIN here as well for point generation
MARGIN = 2.0 

# --- Constants for Point Generation --- 
# Reduce relative distance further
MAX_REL_DIST = CANVAS_SIZE / 6.0 
MIN_OBJ_DEF_DISTANCE = 0.5 # NEW: Minimum distance between points defining an object

class GeoclideanPoint:
    def __init__(self, name, obj_constraints):
        self.name = name
        # Ensure obj_constraints is always a list
        self.obj_constraints = obj_constraints if isinstance(obj_constraints, list) else []
        
    def __str__(self):
        constraints_str = ', '.join(self.obj_constraints) if self.obj_constraints else ''
        return 'Point ' + self.name + '(' + constraints_str + ')'

class GeoclideanObj:
    def __init__(self, name, obj_type, parameters, visibility):
        self.name = name
        self.obj_type = obj_type
        self.parameters = parameters
        self.visibility = visibility
        
    def __str__(self):
        return 'Obj ' + self.name + ': ' + self.obj_type
    
def parse_rule(rule):
    """Parse a rule string into a GeoclideanObj.
    
    Args:
        rule (str): Rule string in format "name = type(p1(c1, c2), p2(c3))"
        
    Returns:
        GeoclideanObj: Parsed geometric object with name, type, and parameters
        
    Raises:
        ValueError: If rule format is invalid or doesn't match expected pattern
    """
    # Strip potential leading/trailing quotes from the whole rule string
    rule = rule.strip().strip('\'\"')
    
    regex = r'(.*) = (.*)\((.*)\((.*)\), (.*)\((.*)\)\)'
    match = re.match(regex, rule)
    
    if not match:
        # Try parsing simple points if the main regex fails (e.g., rules defining only points)
        point_regex = r'(.*)\((.*)\)\''
        point_match = re.match(point_regex, rule)
        if point_match:
            # This indicates a rule defining just a point, which isn't directly handled by GeoclideanObj structure
            # For now, let's raise a specific error or handle as needed.
            # It might be better to pre-process rules to only include object definitions.
            raise ValueError(f"Rule defines only a point, not an object: '{rule}'")
        else:
            raise ValueError(f"Invalid rule format: '{rule}'. Expected format: 'name = type(p1(c1, c2), p2(c3))'")
        
    groups = match.groups()
    raw_name = groups[0].strip().strip('\'\"') # Keep raw name temporarily for visibility check
    obj_type = groups[1].strip()
    
    # Check visibility and determine canonical name (without *)
    visibility = not raw_name.startswith('*')
    name = raw_name[1:] if not visibility else raw_name
    
    # Parse first point parameters
    point_a_name = groups[2].strip()
    point_a_constraints_str = groups[3].strip()
    point_a_constraints = []
    if point_a_constraints_str:  # Only split if there are constraints
        # Strip quotes and * from constraints
        point_a_constraints = [c.strip().strip('\'\"').lstrip('*') for c in point_a_constraints_str.split(',') if c.strip()]
    point_a = GeoclideanPoint(point_a_name, point_a_constraints)
    
    # Parse second point parameters
    point_b_name = groups[4].strip()
    point_b_constraints_str = groups[5].strip()
    point_b_constraints = []
    if point_b_constraints_str:  # Only split if there are constraints
        # Strip quotes and * from constraints
        point_b_constraints = [c.strip().strip('\'\"').lstrip('*') for c in point_b_constraints_str.split(',') if c.strip()]
    point_b = GeoclideanPoint(point_b_name, point_b_constraints)
    
    return GeoclideanObj(name, obj_type, [point_a, point_b], visibility)

def shapely_point_for_point(point, all_shapely_obj, all_shapely_point):
    """Generate a shapely Point based on geometric constraints.
    
    Args:
        point (GeoclideanPoint): Point to generate
        all_shapely_obj (dict): Map of object names to their shapely representations
        all_shapely_point (dict): Map of point names to their shapely representations
        
    Returns:
        shapely.geometry.Point: Generated point meeting constraints
        
    Raises:
        ValueError: If constraints cannot be satisfied or objects don't intersect
    """
    if len(point.obj_constraints) == 0:
        # Random point within canvas bounds (using MARGIN)
        return Point(random.uniform(MARGIN, CANVAS_SIZE - MARGIN), random.uniform(MARGIN, CANVAS_SIZE - MARGIN))
    
    potential_points = []
    first_constraint = True
    
    for constraint_name in point.obj_constraints:
        if constraint_name not in all_shapely_obj and constraint_name not in all_shapely_point:
            raise ValueError(f"Constraint '{constraint_name}' not found in existing objects or points")
            
        # Get the actual shapely object/point from the dictionary
        obj = all_shapely_obj.get(constraint_name) or all_shapely_point.get(constraint_name)
        if obj is None: # Explicit check for None
            raise ValueError(f"Object for constraint '{constraint_name}' is None")
            
        current_constraint_points = []
        # Get points satisfying the current constraint
        if isinstance(obj, Point):
            current_constraint_points = [obj] # If constraint is a point, that's the only option
        elif isinstance(obj, LineString):
            # Sample points along the line
            num_samples = max(2, int(obj.length / 0.5)) # Sample roughly every 0.5 units
            distances = np.linspace(0, obj.length, num_samples)
            current_constraint_points = [obj.interpolate(distance) for distance in distances]
        elif isinstance(obj, Polygon): # Changed from Circle to Polygon
            # Sample points along the exterior boundary (representing a circle)
            exterior_line = obj.exterior
            if exterior_line: # Check if exterior exists
                num_samples = max(20, int(exterior_line.length / 0.5)) # Denser sampling for circles
                distances = np.linspace(0, exterior_line.length, num_samples)
                current_constraint_points = [exterior_line.interpolate(distance) for distance in distances]
            else:
                 # Handle cases where polygon might not have an exterior (e.g., invalid geometry)
                 current_constraint_points = [] 
        else:
             # Handle other potential geometry types if necessary
             print(f"Warning: Unhandled constraint type {type(obj)} for {constraint_name} in shapely_point_for_point")
             current_constraint_points = []

        if not current_constraint_points:
             raise ValueError(f"Could not extract points from constraint object '{constraint_name}' (type: {type(obj)})")

        # Update potential_points based on the constraint
        if first_constraint:
            potential_points = current_constraint_points
            first_constraint = False
        else:
            # Intersect with previous potential points
            new_potential_points = []
            tolerance = 1e-9 # Tolerance for checking if a point lies on the current constraint object
            for p in potential_points:
                # Check if this point p also satisfies the *current* constraint obj
                if obj.distance(p) < tolerance:
                    new_potential_points.append(p)
            potential_points = new_potential_points
            
        if not potential_points:
            # If at any point intersection yields no points, constraints are impossible
            raise ValueError(f"No points satisfy constraints up to '{constraint_name}' for {point.name}")
            
    # If loop finishes and potential_points is still empty (shouldn't happen if logic above is correct, but safety check)
    if not potential_points:
        raise ValueError(f"No points satisfy all constraints for {point.name}")
        
    # Return random point from valid options
    return random.choice(potential_points)
    
def all_interpolated_points_from_obj(obj, sample_distance=0.2):
    interpolated_points = []

    if isinstance(obj, Polygon):
        for i in obj.exterior.coords:
            p = Point(i[0], i[1])
            interpolated_points.append(p)

    elif isinstance(obj, MultiLineString):
        mp = MultiPoint()
        for linestring in obj.geoms:
            for i in np.arange(0, linestring.length, sample_distance):
                s = substring(linestring, i, i+sample_distance)
                mp = mp.union(s.boundary)
        interpolated_points = [p for p in mp.geoms]

    elif isinstance(obj, LineString):
        mp = MultiPoint()
        for i in np.arange(0, obj.length, sample_distance):
            s = substring(obj, i, i+sample_distance)
            mp = mp.union(s.boundary)
        interpolated_points = [p for p in mp.geoms]

    elif isinstance(obj, MultiPoint):
        interpolated_points = [p for p in obj.geoms]

    elif isinstance(obj, Point):
        interpolated_points = [obj]

    return interpolated_points

def render(rules, mark_points=False):
    # Parse to construction
    euclidean_objects = [parse_rule(rule) for rule in rules]

    # Render
    all_shapely_obj, all_shapely_point = {}, {}
    all_viewable_objs = []
    
    current_plot = initial_plot()
    for euc_obj in euclidean_objects:
        # --- Modified Logic --- 
        # 1. Generate and store the base point (p1, p3, p5...) first
        base_point_param = euc_obj.parameters[0]
        point_a_shapely = shapely_point_for_point(base_point_param, all_shapely_obj, all_shapely_point)
        all_shapely_point[base_point_param.name] = point_a_shapely

        # 2. Generate and store the second point (p2, p4, p6...) which might depend on the base point
        size_point_param = euc_obj.parameters[1]
        point_b_shapely = shapely_point_for_point(size_point_param, all_shapely_obj, all_shapely_point)
        all_shapely_point[size_point_param.name] = point_b_shapely
        # --- End Modified Logic --- 

        if euc_obj.obj_type == 'line':
            obj_shapely = action_create_line(point_a_shapely, point_b_shapely)
        if euc_obj.obj_type == 'circle':
            obj_shapely = action_create_circle(point_a_shapely, point_b_shapely)
            
        all_shapely_obj[euc_obj.name] = obj_shapely
            
        if euc_obj.visibility == True:
            current_plot = plot_obj(current_plot, obj_shapely)
            all_viewable_objs.append(obj_shapely)
            
        if mark_points:
            current_plot = plot_point(current_plot, point_a_shapely)
            current_plot = plot_point(current_plot, point_b_shapely)
        
    # Return both the list of visible objects and the dictionary mapping names to all generated shapely objects
    return all_viewable_objs, all_shapely_obj

def numpy_from_plot(ax):
    ax.figure.canvas.draw()
    data = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = ax.figure.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im

def plot_all_except_i(all_viewable_objs, i):
    curr_plot = initial_plot()
    for curr_i, o in enumerate(all_viewable_objs):
        if curr_i == i:
            curr_plot = plot_obj(curr_plot, o, color='white')
        else:
            curr_plot = plot_obj(curr_plot, o)
    curr_plot = numpy_from_plot(curr_plot)
    plt.close()
    return curr_plot

def visibility_test(all_viewable_objs, threshold=400):
    all_plot = initial_plot()
    for o in all_viewable_objs:
        all_plot = plot_obj(all_plot, o)
    all_plot = numpy_from_plot(all_plot)
    plt.close()
    
    for i, o in enumerate(all_viewable_objs):
        curr_plot = plot_all_except_i(all_viewable_objs, i)
        diff = curr_plot[:, :, 0] - all_plot[:, :, 0]
        diff[diff > 1] = 1
        if np.sum(diff) < threshold:
            return False
    return True

def save_steps(all_viewable_objs, dir_name):
    curr_plot = initial_plot()
    for i, o in enumerate(all_viewable_objs):
        curr_plot = plot_obj(curr_plot, o)
        save_plot(dir_name + 'step_' + str(i+1) + '.png')
    plt.close()
    
def save_steps_joint(all_viewable_objs, dir_name, num_steps=3):
    loc = plticker.MultipleLocator(base=1.0)
    fig, plts = plt.subplots(1, num_steps, figsize=(5*num_steps, 5))
    for p in range(num_steps):
        plts[p].xaxis.set_major_locator(loc)
        plts[p].yaxis.set_major_locator(loc)
        plts[p].axis('equal')
        plts[p].axis('off')
        for i in range(p+1):
            plts[p] = plot_obj(plts[p], all_viewable_objs[i])

    save_plot(dir_name)
    plt.close()
            
def generate_concept(rules, mark_points=False, steps_path=None, path=None, show_plots=False):
    i = 0
    while i < 1:
        try:
            all_viewable_objs, all_named_objs = render(rules, mark_points)
            if visibility_test(all_viewable_objs):
                if steps_path:
                    save_steps_joint(all_viewable_objs, steps_path)
                if path:
                    save_plot(path)
                    
                i += 1
                if not show_plots:
                    plt.close()
                
            else:
                plt.close()
        except:
            plt.close()
            continue 
            

def action_create_line(point_a_shapely, point_b_shapely): 
    """Create a shapely LineString, ensuring minimum length.""" 
    if point_a_shapely.distance(point_b_shapely) < MIN_OBJ_DEF_DISTANCE:
        raise ValueError(f"Degenerate line: Points {point_a_shapely.wkt} and {point_b_shapely.wkt} are too close.")
    return LineString([point_a_shapely, point_b_shapely])
    
def action_create_circle(center_shapely, radius_pt_shapely):
    """Create a shapely Polygon representing a circle, ensuring minimum radius."""
    radius = center_shapely.distance(radius_pt_shapely)
    if radius < MIN_OBJ_DEF_DISTANCE:
        # Or should we use a slightly smaller threshold for radius? Using same for now.
        raise ValueError(f"Degenerate circle: Center {center_shapely.wkt} and radius point {radius_pt_shapely.wkt} are too close (radius={radius:.3f}).")
    # Create circle as a buffered point
    return center_shapely.buffer(radius)

def generate_objects_from_concept(rules, mark_points=False, visibility_threshold=400):
    """
    Generates shapely objects from a list of Geoclidean rules, handling dependencies.

    Args:
        rules (list): List of rule strings.
        mark_points (bool): Whether to mark generated points on the plot.
        visibility_threshold (int): Threshold for visibility test.

    Returns:
        tuple: (list_of_visible_shapely_objects, dict_of_all_named_shapely_objects)
               Returns (None, None) if generation fails after max attempts.
    """
    max_generation_passes = len(rules) * 2 # Heuristic limit to prevent infinite loops
    passes = 0
    
    named_objs = {}  # Stores {name: shapely_object} for both points and objects (lines/circles)
    generated_names = set() # Stores names of successfully generated points/objects
    all_viewable_objs = []

    # 1. Parse all rules first
    try:
        parsed_rules = [parse_rule(rule) for rule in rules]
    except ValueError as e:
        print(f"Error parsing rules: {e}")
        return None, None
        
    pending_rules = parsed_rules[:] # Copy list

    # 2. Iterative Generation Loop
    while pending_rules and passes < max_generation_passes:
        passes += 1
        made_progress = False
        rules_processed_in_pass = []

        for i, euc_obj in enumerate(pending_rules):
            # Check if dependencies for both points are met
            deps_met = True
            all_constraints = []
            if euc_obj.parameters[0].obj_constraints:
                all_constraints.extend(euc_obj.parameters[0].obj_constraints)
            if euc_obj.parameters[1].obj_constraints:
                 all_constraints.extend(euc_obj.parameters[1].obj_constraints)
                 
            for constraint in all_constraints:
                if constraint not in generated_names:
                    deps_met = False
                    break
            
            if not deps_met:
                continue # Skip this rule for now, dependencies not ready

            # Dependencies seem met, try to generate
            try:
                point_a_param = euc_obj.parameters[0]
                point_b_param = euc_obj.parameters[1]

                # Generate points only if they haven't been generated before 
                # (points can be defined/used across multiple rules)
                if point_a_param.name not in generated_names:
                    point_a_shapely = shapely_point_for_point(point_a_param, named_objs, named_objs) 
                    named_objs[point_a_param.name] = point_a_shapely
                    generated_names.add(point_a_param.name)
                else:
                     point_a_shapely = named_objs[point_a_param.name]

                if point_b_param.name not in generated_names:
                    point_b_shapely = shapely_point_for_point(point_b_param, named_objs, named_objs) 
                    named_objs[point_b_param.name] = point_b_shapely
                    generated_names.add(point_b_param.name)
                else:
                     point_b_shapely = named_objs[point_b_param.name]
                
                # Create the main object (line or circle)
                if euc_obj.obj_type == 'line':
                    obj_shapely = action_create_line(point_a_shapely, point_b_shapely)
                elif euc_obj.obj_type == 'circle':
                    obj_shapely = action_create_circle(point_a_shapely, point_b_shapely)
                else:
                     raise ValueError(f"Unknown object type: {euc_obj.obj_type}")

                # Store the generated object
                named_objs[euc_obj.name] = obj_shapely
                generated_names.add(euc_obj.name)

                if euc_obj.visibility:
                    all_viewable_objs.append(obj_shapely)

                # Mark as processed in this pass
                rules_processed_in_pass.append(euc_obj)
                made_progress = True

            except ValueError as e:
                # Catch degenerate object errors or point generation errors
                # print(f"  Generation/Validation failed for {euc_obj.name}: {e}. Keeping pending.") # Optional debug
                pass 
            except Exception as e:
                 print(f"  Unexpected error generating {euc_obj.name}: {e}")
                 # Decide whether to keep pending or fail entirely
                 # For now, let's keep it pending but log the error
                 pass

        # Remove successfully processed rules from pending list
        if rules_processed_in_pass:
            pending_rules = [rule for rule in pending_rules if rule not in rules_processed_in_pass]
        
        if not made_progress and pending_rules: 
            # No progress in a full pass, likely circular dependency or persistent error
            print(f"Warning: Could not generate all objects. Remaining rules: {[r.name for r in pending_rules]}")
            # Optionally, you could try one more pass or return failure here
            break # Exit loop to avoid infinite cycling

    # 3. Check if all rules were processed
    if pending_rules:
        print(f"Error: Failed to generate all objects. Unprocessed: {[r.name for r in pending_rules]}")
        # Consider if partially generated results are acceptable or return failure
        # For robustness in the generator script, maybe return what was generated?
        # Let's return failure for now if anything is pending. 
        return None, None 

    # Check overall complexity / spread - prevent overly simple/small images
    if len(all_viewable_objs) < 1 : # Ensure at least one visible object
        # print("Generation resulted in no visible objects. Failing.")
        return None, None
        
    # Combine bounds of all visible objects
    min_x_all, min_y_all, max_x_all, max_y_all = np.inf, np.inf, -np.inf, -np.inf
    for obj in all_viewable_objs:
        b = obj.bounds
        min_x_all = min(min_x_all, b[0])
        min_y_all = min(min_y_all, b[1])
        max_x_all = max(max_x_all, b[2])
        max_y_all = max(max_y_all, b[3])
        
    width = max_x_all - min_x_all
    height = max_y_all - min_y_all
    
    # NEW: Add check for overall size spread
    min_spread = CANVAS_SIZE / 4 # Require the drawing to span at least 1/4 of the canvas width/height
    if width < min_spread or height < min_spread:
        # print(f"Generated image too small (W:{width:.2f}, H:{height:.2f}). Failing.")
        return None, None # Fail generation if overall spread is too small

    return all_viewable_objs, named_objs
            
