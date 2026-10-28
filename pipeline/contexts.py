"""Fixed5-Anchor context contract, shared by generation, SFT preparation and inference.

This module only constructs requests. It does not call an API or train a model.
"""
from copy import deepcopy

TOTAL_FRAMES = 5
DURATION_SECONDS = 2.5

def plan_context(animation_prompt, drawing_style):
    return dict(animation_prompt=animation_prompt, total_frames=TOTAL_FRAMES,
                duration_seconds=DURATION_SECONDS, drawing_style=drawing_style)

def draw_context(animation_prompt, total_plan, index, first_frame=None, previous_frame=None):
    if not isinstance(index,int) or not 1 <= index <= TOTAL_FRAMES:
        raise ValueError('Frame index must be 1..5')
    if index == 1:
        if first_frame is not None or previous_frame is not None:
            raise ValueError('FIRST must not receive future/generated frames')
    else:
        if first_frame is None or first_frame.get('i') != 1:
            raise ValueError('NEXT requires the complete first frame')
        if previous_frame is None or previous_frame.get('i') != index-1:
            raise ValueError('NEXT requires the immediately preceding frame')
        if index == 2 and first_frame != previous_frame:
            raise ValueError('For F2, first_frame and previous_frame must be identical')
    return deepcopy(dict(animation_prompt=animation_prompt, total_plan=total_plan,
        total_frames=TOTAL_FRAMES, duration_seconds=DURATION_SECONDS,
        stage='FIRST' if index == 1 else 'NEXT', current_frame_index=index,
        target_normalized_time=(index-1)/(TOTAL_FRAMES-1),
        target_time_seconds=DURATION_SECONDS*(index-1)/(TOTAL_FRAMES-1),
        first_frame=first_frame, previous_frame=previous_frame))

ANCHOR_INSTRUCTION = '''For NEXT, both first_frame and previous_frame are supplied.
Use first_frame as the persistent reference for object identities, baseline geometry,
dimensions, stroke correspondence and attachment structure. Use previous_frame for
the current pose and motion phase. Advance to the requested time using total_plan;
do not reset moving parts to the first-frame pose. Distinguish an absolute pose
relative to the first frame from an incremental transform of the previous frame.
Within each rigid part, apply one consistent transform to all its path points;
preserve the intended articulation or deformation of non-rigid parts.
Return the complete requested frame in the existing Path3D JSON format.'''

ROTATION_INSTRUCTION = '''When motion includes rotation, choose a consistent pivot,
axis and direction, and transform all endpoints and control points of each rigid
part together. Preserve its dimensions, thickness, shape and attachments; do not
redraw its strokes independently. For a propeller or rotor, preserve blade lengths
and angular spacing about the shared hub, and keep the hub attached to the moving
body. Distinguish local rotation from whole-object motion and absolute orientation
from an increment since the previous frame. Check representative points against
these constraints; keep calculations focused on what the motion requires.'''

def drawing_system(base_system):
    """Shared Fixed5 drawing-system suffix for teacher, SFT and inference.

    Runner integration is still pending; importing this module does not launch jobs.
    """
    return base_system.rstrip() + '\n\n' + ANCHOR_INSTRUCTION + '\n\n' + ROTATION_INSTRUCTION
