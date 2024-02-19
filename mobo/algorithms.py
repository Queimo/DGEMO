from .mobo import MOBO

"""
High-level algorithm specifications by providing config
"""


class DGEMO(MOBO):
    """
    DGEMO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "identity",
        "solver": "discovery",
        "selection": "dgemo",
    }


class TSEMO(MOBO):
    """
    TSEMO
    """

    config = {
        "surrogate": "ts",
        "acquisition": "identity",
        "solver": "nsga2",
        "selection": "hvi",
    }


class USEMO_EI(MOBO):
    """
    USeMO, using EI as acquisition
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "nsga2",
        "selection": "uncertainty",
    }


class MOEAD_EGO(MOBO):
    """
    MOEA/D-EGO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "moead",
        "selection": "moead",
    }


class ParEGO(MOBO):
    """
    ParEGO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "parego",
        "selection": "random",
    }


"""
Define new algorithms here
"""


class Custom(MOBO):
    """
    Totally rely on user arguments to specify each component
    """

    config = None


class PSLbot(MOBO):
    config = {
        # "surrogate": "gp",
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "psl",
        "selection": "hvi",
    }

class RAPSLbot(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "rapsl",
        "selection": "hvi",
    }


class PSL(MOBO):
    config = {
        "surrogate": "gp",
        "acquisition": "identity",
        "solver": "psl",
        "selection": "hvi",
    }

class qNEHVI(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qnehvi",
        "selection": "identity",
    }

class MARS(MOBO):
    config = {
        "surrogate": "botorchgprepeat",
        "acquisition": "identity",
        "solver": "mars",
        "selection": "identity",
    }

class RAqNEHVI(MOBO):
    config = {
        "surrogate": "botorchgprepeat",
        "acquisition": "identity",
        "solver": "raqnehvi",
        "selection": "identity",
    }

class RAqLogNEHVI(MOBO):
    config = {
        "surrogate": "botorchgprepeat",
        "acquisition": "identity",
        "solver": "raqlognehvi",
        "selection": "identity",
    }

class qEHVI(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qehvi",
        "selection": "identity",
    }


def get_algorithm(name):
    """
    Get class of algorithm by name
    """
    algo = {
        "dgemo": DGEMO,
        "tsemo": TSEMO,
        "usemo-ei": USEMO_EI,
        "moead-ego": MOEAD_EGO,
        "parego": ParEGO,
        "custom": Custom,
        "psl": PSL,
        "pslbot": PSLbot, # botorch gp
        "rapslbot": RAPSLbot,
        "qnehvi": qNEHVI,
        "qehvi": qEHVI,
        "mars": MARS,
        "raqnehvi": RAqNEHVI,
        "raqlognehvi": RAqLogNEHVI,
    }
    return algo[name]
