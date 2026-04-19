import proposal as pp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

det_min = np.array([-6, -7, -29.1])
det_max = np.array([6, 7, 29.1])

class in_detector():
    #def __init__(self):
        
    
    def is_outside_box(self, pos, det_min, det_max, buffer=0.0):
        return (
            (pos[0] < det_min[0] - buffer or pos[0] > det_max[0] + buffer) or
            (pos[1] < det_min[1] - buffer or pos[1] > det_max[1] + buffer) or
            (pos[2] < det_min[2] - buffer or pos[2] > det_max[2] + buffer)
        )
    
    def ray_box_path_length(self, origin, direction, box_min, box_max):
        tmin = -np.inf
        tmax = np.inf

        for i in range(3):  # x, y, z
            if direction[i] != 0:
                t1 = (box_min[i] - origin[i]) / direction[i]
                t2 = (box_max[i] - origin[i]) / direction[i]
                t_near = min(t1, t2)
                t_far = max(t1, t2)
                tmin = max(tmin, t_near)
                tmax = min(tmax, t_far)
            elif not (box_min[i] <= origin[i] <= box_max[i]):
                return False, 0.0, 0.0  # Ray is parallel and outside the box

        if tmax > max(tmin, 0):
            path_length = tmax - max(tmin, 0)
            entry_path = max(tmin, 0)
            return True, path_length, entry_path
        else:
            return False, 0.0, 0.0

        
class get_throughgoing_events(in_detector):
    def __init__(self, json_file, muon_file):
        self.prop_env = json_file

        self.particle = pp.particle.MuMinusDef()
        self.propagator = pp.Propagator(self.particle, self.prop_env)

        self.events =[]

        with open(muon_file, "r") as f:
            chunks = f.read().strip().split("\n\n")  # split each event
            for chunk in chunks:
                event = json.loads(chunk)
                self.events.append(event)
        
    def sort_events(self, type):
        with open(f"propagated/throughgoing_propagated_{type}.txt", "w") as f, open("stochastic_losses.txt", "w") as f_losses:
            for i, event in enumerate(self.events):
                initial_state = pp.particle.ParticleState()
                initial_state.energy = np.squeeze(event["primary_momentum"])[0] * 1000
                vertex = np.squeeze(event["vertex"]) * 100
                initial_state.position = pp.Cartesian3D(*vertex)

                p = np.squeeze(event["primary_momentum"])
                direction = p[1:]
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue  # skip invalid direction
                direction /= norm
                initial_state.direction = pp.Cartesian3D(*direction)

                try:
                    # Keep only throughgoing events
                    did_intersect, expected_path, dist = in_detector.ray_box_path_length(self, vertex, direction,
                                                                                         det_min * 100, det_max * 100)
            
                    if (did_intersect):
                        secondaries = self.propagator.propagate(initial_state, max_distance=dist)
                        final_state = secondaries.final_state()

                        final_pos = np.array([final_state.position.x, final_state.position.y, final_state.position.z])
            
            
                        if (
                            self.is_outside_box(vertex / 100, det_min, det_max) and
                            final_state.propagated_distance >= dist
                        ):
                            line = (
                                f"{event['event_weight']}  "
                                f"{final_state.type}  "
                                f"[{final_pos[0]:.5f}, {final_pos[1]:.5f}, {final_pos[2]:.5f}]  "
                                f"[{final_state.direction.x:.3f}, {final_state.direction.y:.3f}, {final_state.direction.z:.3f}]  "
                                f"{final_state.energy:.6f}  "
                                f"{final_state.time:.6e}  "
                                f"{final_state.propagated_distance:.6f}" # in cm!
                            )
                            f.write(line + "\n")
                
                            losses = secondaries.stochastic_losses()
                            for loss in losses:
                                loss_line = (
                                    f"{loss.energy}  "
                                    f"{loss.parent_particle_energy}  "
                                    f"[{loss.position.x:.5f}, {loss.position.y:.5f}, {loss.position.z:.5f}]  "
                                    f"{pp.particle.Interaction_Type(loss.type).name}  "
                                )
                                f_losses.write(loss_line + "\n")
                            f_losses.write("\n")
                except RuntimeError as e:
                   print(f"RuntimeError on event {i}: {e}")