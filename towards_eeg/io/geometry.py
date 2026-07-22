#!/usr/bin/env python3
"""
geometry.py -- the abstract geometry record that I-15 and I-18 are written
against.

WHY THIS EXISTS
---------------
Invariants I-15 and I-18 are statements about morphology geometry:

    I-15  for all nu: exactly one section of somatic calibre exists AND is
          the topology root                     (catches the 489469961 root failure)
    I-18  for all nu: the somatic section is identified by DIAMETER AND
          POSITION, not by name, and the name-based and geometry-based
          identifications agree                 (catches O8, label precedence)

Both need diameters, positions and topology. But TEEG_00 section 1 is explicit:
the morphology .hoc files are NOT attached to an S0 chat, S0 does not touch
morphology, and they belong to S1.

The resolution, decided 21 July 2026: define the minimal record the invariants
actually consume, write the invariants against THAT, and test them against
synthetic records. The functions are real and exercised; no morphology file
enters S0. S1 supplies a reader that produces this record from a .hoc, and the
invariants do not change when it does.

WHAT THE RECORD IS NOT
----------------------
It is not a morphology representation. It carries only what I-15 and I-18
read: per section, an identifier, the emitted name, a representative diameter,
a position, and a parent. It cannot represent a cable, a segment or a
mechanism, and it should not grow to. Anything needing more is S1's Cell.

Standard library only, ASCII source, Python 3.8+.
"""

import math

__all__ = ["Section", "Geometry", "SOMATIC_CALIBRE_RATIO", "GeometryError"]


class GeometryError(ValueError):
    """A geometry record is malformed."""


# The threshold separating somatic calibre from neuritic calibre, as a
# multiple of the median section diameter.
#
# THIS IS A DECLARED PARAMETER, NOT A CONTRACT CONSTANT. TEEG_02 C8.1 gives
# the evidence -- in neuron_489469961_aligned.hoc the soma is 11.22 um against
# a median of 0.40 um over all sections, a ratio of about 28 -- but states no
# threshold. 5.0 sits an order of magnitude below the observed ratio and well
# above any plausible dendritic trunk, and it is exposed as an argument so
# that a bank which disagrees produces an argument rather than a silent
# reclassification. If S1 measures a distribution that makes this wrong, it is
# a parameter change, not a code change.
SOMATIC_CALIBRE_RATIO = 5.0


class Section(object):
    """One NEURON section, reduced to what I-15 and I-18 read.

    Parameters
    ----------
    sid : hashable
        Section identifier, unique within one Geometry.
    name : str
        The name as EMITTED in the .hoc, e.g. 'soma[0]' or -- in a bank
        exhibiting O8 -- 'exc_syn[0]'. Deliberately kept separate from
        (cls, dom): the whole point of I-18 is that the name may be wrong.
    diameter_um : float
        Representative diameter, in um. Strictly positive.
    position_um : tuple of three floats
        Centroid in the aligned cell frame, in um.
    parent : hashable or None
        The sid of the parent section, or None for the topology root.
        Exactly one section in a well-formed Geometry has parent None.
    """

    __slots__ = ("sid", "name", "diameter_um", "position_um", "parent")

    def __init__(self, sid, name, diameter_um, position_um, parent):
        if diameter_um is None or diameter_um <= 0:
            raise GeometryError("section %r has non-positive diameter %r"
                                % (sid, diameter_um))
        if len(position_um) != 3:
            raise GeometryError("section %r position is not 3-dimensional"
                                % (sid,))
        self.sid = sid
        self.name = name
        self.diameter_um = float(diameter_um)
        self.position_um = tuple(float(c) for c in position_um)
        self.parent = parent

    def distance_from(self, origin_um=(0.0, 0.0, 0.0)):
        """Euclidean distance of this section's centroid from a point, in um."""
        return math.sqrt(sum((a - b) ** 2
                             for a, b in zip(self.position_um, origin_um)))

    def __repr__(self):
        return ("Section(sid=%r, name=%r, diameter_um=%r, parent=%r)"
                % (self.sid, self.name, self.diameter_um, self.parent))


class Geometry(object):
    """The sections of one morphology, with the alignment origin.

    Parameters
    ----------
    nid : int
        nu, the H01 morphology identifier. Carried so that an invariant
        failure names the morphology it failed on.
    sections : sequence of Section
    alignment_origin_um : tuple of three floats
        The origin of the aligned cell frame. The soma is expected to sit at
        it -- TEEG_02 C8.1 records the soma of 489469961 as "centred exactly
        at the alignment origin" -- and I-18 uses that as the positional half
        of its geometry-based identification.
    """

    def __init__(self, nid, sections, alignment_origin_um=(0.0, 0.0, 0.0)):
        self.nid = nid
        self.sections = list(sections)
        if not self.sections:
            raise GeometryError("morphology %r has no sections" % (nid,))
        seen = set()
        for sec in self.sections:
            if sec.sid in seen:
                raise GeometryError("morphology %r repeats section id %r"
                                    % (nid, sec.sid))
            seen.add(sec.sid)
        for sec in self.sections:
            if sec.parent is not None and sec.parent not in seen:
                raise GeometryError("morphology %r: section %r names a parent "
                                    "%r that does not exist"
                                    % (nid, sec.sid, sec.parent))
        if len(alignment_origin_um) != 3:
            raise GeometryError("alignment origin is not 3-dimensional")
        self.alignment_origin_um = tuple(float(c) for c in alignment_origin_um)

    # -- the two identifications that I-18 compares -------------------------

    def median_diameter_um(self):
        d = sorted(sec.diameter_um for sec in self.sections)
        n = len(d)
        if n % 2:
            return d[n // 2]
        return 0.5 * (d[n // 2 - 1] + d[n // 2])

    def somatic_calibre_sections(self, ratio=SOMATIC_CALIBRE_RATIO):
        """Sections whose diameter is at least `ratio` times the median.

        I-15 asserts this set has exactly one element. Returned as a list
        rather than an optional single section precisely so that the "more
        than one" failure is expressible.
        """
        threshold = ratio * self.median_diameter_um()
        return [sec for sec in self.sections if sec.diameter_um >= threshold]

    def identify_soma_by_geometry(self, ratio=SOMATIC_CALIBRE_RATIO):
        """The somatic section per diameter AND position. May be None.

        Both halves are required by I-18. Diameter alone would accept a thick
        proximal trunk; position alone would accept whatever section happens
        to lie nearest the origin in a morphology whose soma was mislabelled
        away entirely.
        """
        candidates = self.somatic_calibre_sections(ratio=ratio)
        if not candidates:
            return None
        return min(candidates,
                   key=lambda s: s.distance_from(self.alignment_origin_um))

    def identify_soma_by_name(self, token="soma"):
        """Sections whose emitted name contains `token`.

        A list, not a section: in a bank exhibiting O8 this returns EMPTY for
        a morphology whose soma was emitted as exc_syn[0], and that emptiness
        is the finding, not an error to be smoothed over.
        """
        return [sec for sec in self.sections if token in sec.name]

    def root(self):
        """The section with no parent, or None if there is not exactly one."""
        roots = [sec for sec in self.sections if sec.parent is None]
        return roots[0] if len(roots) == 1 else None

    def __repr__(self):
        return "Geometry(nid=%r, n_sections=%d)" % (self.nid, len(self.sections))
