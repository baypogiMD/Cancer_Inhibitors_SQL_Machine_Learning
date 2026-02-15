SELECT
    activity_label,
    COUNT(*) AS count
FROM inhibitors
GROUP BY activity_label;
