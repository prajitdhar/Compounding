"""Equal-decade context ranking shared by corpus-specific vocabulary stages.

Callers supply one independent occurrence-count row per context key and decade,
after their corpus-specific POS filtering and lemma mapping. Missing context
keys contribute zero in other decades. Surface forms, coverage exports and
manifest handling belong to the calling project.
"""


def _identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def equal_decade_ranking_query(source, keys, *, total_count="tokens",
                              observed_decades=None, ordered_sum=True):
    """Return DuckDB SQL ranking aggregated ``decade,count`` occurrence rows.

    ``keys`` specify both context identity and the lexical tie-break order.
    By default, the denominator counts decades in the supplied eligible rows.
    ``observed_decades`` explicitly includes other observed corpus decades that
    have no eligible context; it must be positive. This distinction preserves
    each corpus adapter's established score population.

    Ordered summation gives reproducible floating-point scores. A caller can
    retain its existing unordered summation policy with ``ordered_sum=False``.
    The query does not truncate ranks or filter counts; callers retain their
    own top-k, zero-count and surface-projection policies.
    """
    keys = tuple(keys)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Context keys must be nonempty and unique")
    if observed_decades is not None and (
            not isinstance(observed_decades, int) or observed_decades < 1):
        raise ValueError("The observed-decade denominator must be a positive integer")
    key_list = ", ".join(_identifier(key) for key in keys)
    selected_keys = ", ".join("d." + _identifier(key) for key in keys)
    denominator = ("SELECT COUNT(*)::DOUBLE AS n_decades FROM decade_totals"
                   if observed_decades is None else
                   f"SELECT {observed_decades}::DOUBLE AS n_decades")
    summation_order = " ORDER BY d.decade" if ordered_sum else ""
    total_count = _identifier(total_count)
    return f"""WITH decade_totals AS (
        SELECT decade, SUM(count)::DOUBLE AS decade_total
        FROM {_identifier(source)} GROUP BY decade
    ), decade_count AS (
        {denominator}
    ), scores AS (
        SELECT {selected_keys},
               SUM(d.count / t.decade_total{summation_order}) / dc.n_decades AS score,
               COUNT(*)::BIGINT AS decades_observed,
               SUM(d.count)::BIGINT AS {total_count}
        FROM {_identifier(source)} d JOIN decade_totals t USING (decade)
        CROSS JOIN decade_count dc
        GROUP BY {selected_keys}, dc.n_decades
    )
    SELECT {key_list}, score, decades_observed, {total_count},
           ROW_NUMBER() OVER (ORDER BY score DESC, {key_list}) AS rank
    FROM scores"""
