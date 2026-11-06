"""Finite opaque planar patches, first-order geometric optics only."""
from dataclasses import dataclass

import numpy as np

EPS = 1e-8


@dataclass
class Panel:
    name: str
    center: np.ndarray
    normal: np.ndarray
    width: float
    height: float

    @property
    def tangent(self):
        return np.cross(np.array([0., 0., 1.]), self.normal)

    def margins(self, point):
        delta = point-self.center
        return np.array([self.width/2-abs(delta@self.tangent),
                         self.height/2-abs(delta[2])])

    def vertices(self):
        return [self.center+a*self.width/2*self.tangent+np.array([0., 0., b*self.height/2])
                for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]]

    def as_dict(self):
        return {'name': self.name, 'center_m': self.center.tolist(), 'normal': self.normal.tolist(),
                'width_m': self.width, 'height_m': self.height,
                'vertices_m': [x.tolist() for x in self.vertices()]}


def segment_hit(a, b, panel):
    da, db = (a-panel.center)@panel.normal, (b-panel.center)@panel.normal
    denominator = db-da
    if abs(denominator) < EPS:
        return {'blocked': False, 'degenerate': bool(abs(da) < EPS), 'edge': False}
    t = -da/denominator
    if t <= EPS or t >= 1-EPS:
        return {'blocked': False, 'degenerate': bool(abs(da) < EPS or abs(db) < EPS), 'edge': False}
    point = a+t*(b-a)
    margin = panel.margins(point)
    return {'blocked': bool(np.min(margin) >= -EPS), 'degenerate': False,
            'edge': bool(np.min(abs(margin)) < EPS and np.min(margin) >= -EPS),
            'intersection_m': point.tolist(), 'fraction': float(t)}


def reflection(a, b, panel, panels):
    da, db = (a-panel.center)@panel.normal, (b-panel.center)@panel.normal
    if da*db <= EPS:
        return {'status': 'opposite_side_or_on_plane', 'panel': panel.name}
    image = a-2*da*panel.normal
    denominator = (b-image)@panel.normal
    if abs(denominator) < EPS:
        return {'status': 'degenerate', 'panel': panel.name}
    t = ((panel.center-image)@panel.normal)/denominator
    point = image+t*(b-image)
    margin = panel.margins(point)
    if not 0 < t < 1 or np.min(margin) < -EPS:
        return {'status': 'outside_patch', 'panel': panel.name, 'point_m': point.tolist()}
    if np.min(margin) <= EPS:
        return {'status': 'edge_degenerate', 'panel': panel.name, 'point_m': point.tolist()}
    blockers, degeneracy = [], False
    for other in panels:
        if other.name == panel.name:
            continue
        for first,last in [(a,point),(point,b)]:
            hit = segment_hit(first,last,other)
            degeneracy |= hit['degenerate'] or hit['edge']
            if hit['blocked']:
                blockers.append(other.name)
    if blockers or degeneracy:
        return {'status': 'occluded' if blockers else 'degenerate', 'panel': panel.name,
                'point_m': point.tolist(), 'blockers': blockers, 'degenerate': degeneracy}
    incoming = (point-a)/np.linalg.norm(point-a)
    outgoing = (b-point)/np.linalg.norm(b-point)
    length = np.linalg.norm(a-point)+np.linalg.norm(point-b)
    return {'status': 'visible', 'kind': 'reflection', 'panel': panel.name,
            'point_m': point.tolist(), 'length_m': float(length),
            'image_length_error_m': float(abs(length-np.linalg.norm(image-b))),
            'reflection_law_error': float(np.linalg.norm(incoming-2*(incoming@panel.normal)*panel.normal-outgoing)),
            'margin_m': margin.tolist()}


def trace(a, b, panels):
    intersections = {p.name:segment_hit(a,b,p) for p in panels}
    blocked = [k for k,v in intersections.items() if v['blocked']]
    degenerate = any(v['degenerate'] or v['edge'] for v in intersections.values())
    paths = []
    if not blocked and not degenerate:
        paths.append({'kind': 'direct', 'length_m': float(np.linalg.norm(a-b)), 'panel': None})
    reflected = [reflection(a,b,p,panels) for p in panels]
    paths.extend(r for r in reflected if r['status']=='visible')
    return {'direct_blockers': blocked, 'direct_intersections': intersections,
            'degenerate': degenerate or any('degenerate' in r['status'] or r.get('degenerate',False) for r in reflected),
            'reflection_attempts': reflected, 'paths': paths}


def fixed_panels():
    return [Panel('A',np.array([-250.,0.,30.]),np.array([1.,0.,0.]),400.,60.),
            Panel('B',np.array([-50.,-550.,30.]),np.array([0.,1.,0.]),500.,60.)]


def unit_tests():
    p = Panel('T',np.array([0.,0.,1.]),np.array([1.,0.,0.]),4.,2.)
    a,b = np.array([-2.,-1.,1.]),np.array([-2.,1.,1.])
    r = reflection(a,b,p,[p])
    blocker = Panel('Q',np.array([-1.,-.5,1.]),np.array([0.,1.,0.]),1.,2.)
    checks = {
        'empty_scene_direct': len(trace(a,b,[])['paths'])==1,
        'finite_center_blocks': segment_hit(np.array([-2.,0.,1.]),np.array([2.,0.,1.]),p)['blocked'],
        'above_patch_clear': not segment_hit(np.array([-2.,0.,3.]),np.array([2.,0.,3.]),p)['blocked'],
        'beside_patch_clear': not segment_hit(np.array([-2.,3.,1.]),np.array([2.,3.,1.]),p)['blocked'],
        'edge_contact_reported': segment_hit(np.array([-2.,2.,1.]),np.array([2.,2.,1.]),p)['edge'],
        'coplanar_reported': segment_hit(np.array([0.,-1.,1.]),np.array([0.,1.,1.]),p)['degenerate'],
        'same_side_reflection': r['status']=='visible' and np.linalg.norm(np.array(r['point_m'])-[0.,0.,1.])<1e-10,
        'specular_law': r['reflection_law_error']<1e-12 and r['image_length_error_m']<1e-12,
        'opposite_side_not_reflected': reflection(a,np.array([2.,1.,1.]),p,[p])['status']=='opposite_side_or_on_plane',
        'other_patch_occludes_leg': reflection(a,b,p,[p,blocker])['status']=='occluded',
        'reciprocal_reflection': abs(reflection(b,a,p,[p])['length_m']-r['length_m'])<1e-12,
    }
    return {k:bool(v) for k,v in checks.items()}
