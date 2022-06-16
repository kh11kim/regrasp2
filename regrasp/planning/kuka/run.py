from math import floor

from pyparsing import CaselessKeyword
from regrasp.pddlstream.examples.pybullet.utils.pybullet_tools.utils import SINK_URDF
from regrasp.utils.scene_maker import BulletSceneMaker
from regrasp.utils.world import BulletWorld
from regrasp.utils.transform import Transform
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object, negate_test

DRAKE_PATH = 'regrasp/pddlstream/examples/pybullet/utils/models/drake/iiwa_description/urdf/iiwa14_polytope_collision.urdf'
SMALL_BLOCK_URDF = 'regrasp/pddlstream/examples/pybullet/utils/models/drake/objects/block_for_pick_and_place.urdf'
BLOCK_URDF = 'regrasp/pddlstream/examples/pybullet/utils/models/drake/objects/block_for_pick_and_place_mid_size.urdf'
SINK_URDF = 'regrasp/pddlstream/examples/pybullet/utils/models/sink.urdf'
STOVE_URDF = 'regrasp/pddlstream/examples/pybullet/utils/models/stove.urdf'

def load_world():
    world = BulletWorld(gui=True)
    sm = BulletSceneMaker(world)
    robot = world.load_urdf("kuka", DRAKE_PATH)
    floor = sm.create_plane()
    sink = world.load_urdf("sink", SINK_URDF, pose=Transform(translation=[0,0.5,0]), use_fixed_base=True)
    stove = world.load_urdf("stove", STOVE_URDF, pose=Transform(translation=[0,-0.5,0]), use_fixed_base=True)
    #movable
    celery = world.load_urdf("celery", BLOCK_URDF, pose=Transform(translation=[0.5, 0, 0]))
    celery.set_stable_z()
    radish = world.load_urdf("radish", SMALL_BLOCK_URDF, pose=Transform(translation=[-0.5, 0, 0]))
    radish.set_stable_z()
    
    movable = dict(
        celery=celery,
        radish=radish
    )
    return world, robot, movable

def sample_placement():
    pass
def main():
    world, robot, movable = load_world()
    world.save_state()
    
    #pddlstream
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))

    input()


if __name__ == "__main__":
    main()