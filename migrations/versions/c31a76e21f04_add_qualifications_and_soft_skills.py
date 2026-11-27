"""add qualifications and soft skills to jobs

Revision ID: c31a76e21f04
Revises: b842d50d4d68
Create Date: 2026-08-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c31a76e21f04"
down_revision: str | Sequence[str] | None = "b842d50d4d68"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add non-technical job requirement categories."""

    op.add_column(
        "jobs",
        sa.Column(
            "qualifications",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "jobs",
        sa.Column(
            "soft_skills",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    """Remove non-technical job requirement categories."""

    op.drop_column("jobs", "soft_skills")
    op.drop_column("jobs", "qualifications")
