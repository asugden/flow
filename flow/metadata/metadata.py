"""Experimental metadata."""
from . import parser



class MissingParentError(Exception):
    pass


class AlreadyPresentError(Exception):
    pass


def dataframe(
        mice=None, dates=None, runs=None, run_types=None, tags=None,
        photometry=None, sort=False):
    """Return metadata as a DataFrame, optionally filtering on any columns.

    All parameters are optional and if passed will be used to filter the
    columns of the resulting dataframe.

    Parameters
    ----------
    mice : list of str
    dates : list of int
    runs : list of int
    run_types : list of str
    tags : list of str
    photometry : list of str
    sort : bool
        If True, sort rows by (mouse, date, run).

    Returns
    -------
    pd.DataFrame
        Columns=('mouse', 'date', 'run', 'run_type', 'tags', 'photometry')

    """
    df = parser.meta_df()

    if mice is not None:
        df = df[df.mouse.isin(mice)]
    if dates is not None:
        df = df[df.date.isin(dates)]
    if runs is not None:
        df = df[df.run.isin(runs)]
    if run_types is not None:
        df = df[df.run_type.isin(run_types)]
    if tags is not None:
        df = df[df.mouse_tags.apply(
            lambda x: any(tag in x for tag in tags))]
    if photometry is not None:
        df = df[df.photometry.apply(
            lambda x: any(tag in x for tag in photometry))]

    if sort:
        df = df.sort_values(by=['mouse', 'date', 'run']).reset_index(drop=True)

    return df


def add_mouse(mouse, tags=None, overwrite=False):
    """Add a mouse to the metadata.

    Parameters
    ----------
    mouse : str
    tags : list of str, optional
    overwrite : bool

    """
    metadata = parser.meta_dict()
    existing_mouse = [m for m in metadata['mice'] if m['name'] == mouse]
    assert len(existing_mouse) <= 1
    if len(existing_mouse) == 1:
        if not overwrite:
            raise AlreadyPresentError(
                'Mouse already present in metadata: {}'.format(mouse))
        metadata['mice'].remove(existing_mouse[0])
    mouse_dict = {'name': mouse,
                  'dates': []}
    if tags is not None:
        mouse_dict['tags'] = tags
    metadata['mice'].append(mouse_dict)
    parser.save(metadata)
    parser.meta_dict(reload_=True)


def add_date(mouse, date, tags=None, photometry=None, overwrite=False):
    """Add a date to the metadata.

    Mouse must already exist.

    Parameters
    ----------
    mouse : str
    date : int
    tags : list of str, optional
    photometry : list of str, optional
    overwrite : bool

    """
    metadata = parser.meta_dict()
    existing_mouse = [m for m in metadata['mice'] if m['name'] == mouse]
    if len(existing_mouse) == 0:
        raise MissingParentError('Must first add mouse: {}'.format(mouse))
    assert len(existing_mouse) == 1
    mouse_dict = existing_mouse[0]
    metadata['mice'].remove(mouse_dict)

    existing_date = [d for d in mouse_dict['dates'] if d['date'] == date]
    assert len(existing_date) <= 1
    if len(existing_date) == 1:
        if not overwrite:
            raise AlreadyPresentError(
                'Date already exists in metadata: {}-{}'.format(mouse, date))
        mouse_dict['dates'].remove(existing_date[0])

    date_dict = {'date': date,
                 'runs': []}
    if tags is not None:
        date_dict['date_tags'] = tags
    if photometry is not None:
        date_dict['photometry'] = photometry

    mouse_dict['dates'].append(date_dict)
    metadata['mice'].append(mouse_dict)

    parser.save(metadata)
    parser.meta_dict(reload_=True)


def add_run(
        mouse, date, run, run_type, tags=None, overwrite=False):
    """Add a run to the metadata.

    Parameters
    ----------
    mouse : str
    date : int
    run : int
    run_type : str
    tags : list of str, optional
    overwrite : bool

    """
    metadata = parser.meta_dict()
    existing_mouse = [m for m in metadata['mice'] if m['name'] == mouse]
    if len(existing_mouse) == 0:
        raise MissingParentError('Must first add mouse: {}'.format(mouse))
    assert len(existing_mouse) == 1
    mouse_dict = existing_mouse[0]
    metadata['mice'].remove(mouse_dict)

    existing_date = [d for d in mouse_dict['dates'] if d['date'] == date]
    if len(existing_date) == 0:
        raise MissingParentError('Must first add date: {}-{}'.format(mouse, date))
    assert len(existing_date) == 1
    date_dict = existing_date[0]
    mouse_dict['dates'].remove(date_dict)

    existing_run = [r for r in date_dict['runs'] if r['run'] == run]
    assert len(existing_run) <= 1
    if len(existing_run) == 1:
        if not overwrite:
            raise AlreadyPresentError(
                'Run already exists in metadata: {}-{}-{}'.format(
                    mouse, date, run))
        date_dict['runs'].remove(existing_run[0])

    run_dict = {'run': run,
                'run_type': run_type}
    if tags is not None:
        run_dict['run_tags'] = tags

    date_dict['runs'].append(run_dict)
    mouse_dict['dates'].append(date_dict)
    metadata['mice'].append(mouse_dict)

    parser.save(metadata)
    parser.meta_dict(reload_=True)
