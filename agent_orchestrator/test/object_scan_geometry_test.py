import math
import unittest

from targeted_scan_geometry import generate_object_arc_angles, object_arc_pose


class ObjectScanGeometryTest(unittest.TestCase):
    def test_arc_sweeps_full_orbit_from_minus_180_to_plus_180(self):
        angles = generate_object_arc_angles(
            start_angle=math.radians(-180.0),
            arc_degrees=360.0,
            viewpoints=18,
            direction=1,
        )
        self.assertEqual(len(angles), 18)
        self.assertAlmostEqual(angles[0], math.radians(-180.0))
        self.assertAlmostEqual(angles[-1], math.radians(180.0))
        increments = [right - left for left, right in zip(angles, angles[1:])]
        self.assertTrue(all(value > 0.0 for value in increments))
        self.assertTrue(all(math.isclose(value, increments[0]) for value in increments))

    def test_recentered_pose_preserves_horizontal_radius_and_height(self):
        pose = object_arc_pose(
            target_position=(0.2, -0.1, 0.08),
            bearing=1.2,
            radius=0.25,
            height_offset=0.3,
        )
        horizontal_distance = math.hypot(pose.x - 0.2, pose.y + 0.1)
        self.assertAlmostEqual(horizontal_distance, 0.25)
        self.assertAlmostEqual(pose.z, 0.38)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_object_arc_angles(
                start_angle=0.0,
                arc_degrees=361.0,
                viewpoints=13,
                direction=1,
            )
        with self.assertRaises(ValueError):
            object_arc_pose(
                target_position=(0.0, 0.0, 0.0),
                bearing=0.0,
                radius=0.0,
                height_offset=0.3,
            )


if __name__ == "__main__":
    unittest.main()
